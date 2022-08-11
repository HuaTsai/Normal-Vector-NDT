#include <common/eigen_utils.h>
#include <ndt/costs.h>
#include <ndt/matcher.h>
#include <ndt/opt.h>
#include <ndt/orj.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree.h>

NDTMatcher NDTMatcher::GetIter(std::unordered_set<Options> options,
                               std::vector<double> cell_sizes,
                               double d2) {
  options.insert(Options::kIterative);
  return NDTMatcher(options, cell_sizes, 0, d2, 0.005);
}

NDTMatcher NDTMatcher::GetBasic(std::unordered_set<Options> options,
                                double cell_size,
                                double d2) {
  if (options.count(Options::kIterative)) {
    std::cerr << __FUNCTION__ << ": invalid options\n";
    std::exit(1);
  }
  return NDTMatcher(options, {}, cell_size, d2, 0.005);
}

NDTMatcher::NDTMatcher(std::unordered_set<Options> options,
                       std::vector<double> cell_sizes,
                       double cell_size,
                       double d2,
                       double intrinsic)
    : Matcher(options),
      cell_sizes_(cell_sizes),
      cell_size_(cell_size),
      d2_(d2),
      intrinsic_(intrinsic) {
  if ((HasOption(Options::kNDT) && HasOption(Options::kNVNDT)) ||
      (HasOption(Options::k1to1) && HasOption(Options::k1ton))) {
    std::cerr << __FUNCTION__ << ": invalid options\n";
    std::exit(1);
  }
}

Eigen::Affine3d NDTMatcher::AlignImpl(const Eigen::Affine3d &guess) {
  // the value here may be crucial!
  Eigen::Matrix3d pcov = Eigen::Matrix3d::Identity() * intrinsic_;
  timer_.ProcedureStart(timer_.kNDT);
  tmap_ = std::make_shared<NMap>(cell_size_);
  if (HasOption(Options::kPointCov))
    tmap_->LoadPointsWithCovariances(tpts_, pcov);
  else
    tmap_->LoadPoints(tpts_);
  timer_.ProcedureFinish();

  timer_.ProcedureStart(timer_.kNDT);
  smap_ = std::make_shared<NMap>(cell_size_);
  if (HasOption(Options::kPointCov))
    smap_->LoadPointsWithCovariances(spts_, pcov);
  else
    smap_->LoadPoints(spts_);
  timer_.ProcedureFinish();

  for (const auto &cell : *smap_) {
    Eigen::Vector3d tmp1;
    Eigen::Matrix3d tmp2;
    timer_.ProcedureStart(timer_.kNormal);
    ComputeEvalEvec(cell.second.GetCov(), tmp1, tmp2);
    timer_.ProcedureFinish();
  }
  for (const auto &cell : *tmap_) {
    Eigen::Vector3d tmp1;
    Eigen::Matrix3d tmp2;
    timer_.ProcedureStart(timer_.kNormal);
    ComputeEvalEvec(cell.second.GetCov(), tmp1, tmp2);
    timer_.ProcedureFinish();
  }

  auto cur_tf = guess;
  bool converge = false;

  std::vector<Eigen::Affine3d> tfs;
  tfs.push_back(cur_tf);

  while (!converge) {
    timer_.ProcedureStart(timer_.kBuild);
    auto next = smap_->TransformCells(cur_tf);
    std::vector<Eigen::Vector3d> ups, uqs;
    std::vector<Eigen::Matrix3d> cps, cqs;
    std::vector<Eigen::Vector3d> nps, nqs;
    for (const auto &cellp : next) {
      if (!cellp.GetHasGaussian()) continue;
      if (HasOption(Options::k1to1)) {
        auto cellq = tmap_->SearchNearestCell(cellp.GetMean());
        if (!cellq.GetHasGaussian()) continue;
        Eigen::Vector3d np = cellp.GetNormal();
        Eigen::Vector3d nq = cellq.GetNormal();
        if (np.dot(nq) < 0) np = -np;
        nps.push_back(np);
        nqs.push_back(nq);
        ups.push_back(cellp.GetMean());
        cps.push_back(cellp.GetCov());
        uqs.push_back(cellq.GetMean());
        cqs.push_back(cellq.GetCov());
      } else if (HasOption(Options::k1ton)) {
        auto cellqs = tmap_->SearchCellsInRadius(cellp.GetMean(), cell_size_);
        for (auto c : cellqs) {
          const Cell &cellq = c.get();
          if (!cellq.GetHasGaussian()) continue;
          Eigen::Vector3d np = cellp.GetNormal();
          Eigen::Vector3d nq = cellq.GetNormal();
          if (np.dot(nq) < 0) np = -np;
          nps.push_back(np);
          nqs.push_back(nq);
          ups.push_back(cellp.GetMean());
          cps.push_back(cellp.GetCov());
          uqs.push_back(cellq.GetMean());
          cqs.push_back(cellq.GetCov());
        }
      }
    }

    if (!HasOption(Options::kNoReject)) {
      Orj orj(ups.size());
      // orj.RangeRejection(ups, uqs, Rejection::kBoth, {cell_size_});
      orj.RangeRejection(ups, uqs, Rejection::kThreshold, {cell_size_});
      // orj.AngleRejection(nps, nqs, Rejection::kThreshold, {1});
      orj.RetainIndices(ups, uqs, cps, cqs, nps, nqs);
    }

    Options type = Options::k3D;
    if (HasOption(Options::kAnalytic)) type = Options::kAnalytic;

    Optimizer opt(type);
    opt.set_cur_tf3(cur_tf);
    corres_ = ups.size();
    if (HasOption(Options::kNDT)) {
      if (HasOption(Options::kAnalytic))
        opt.BuildProblem(new D2DNDTCost(ups, cps, uqs, cqs, d2_));
      else
        opt.BuildProblem(D2DNDTCostAuto::Create(ups, cps, uqs, cqs, d2_));
    } else if (HasOption(Options::kNVNDT)) {
      if (HasOption(Options::kAnalytic))
        opt.BuildProblem(new NVNDTCost(ups, cps, nps, uqs, cqs, nqs, d2_));
      else
        opt.BuildProblem(
            NVNDTCostAuto::Create(ups, cps, nps, uqs, cqs, nqs, d2_));
    }
    timer_.ProcedureFinish();

    timer_.ProcedureStart(timer_.kOptimize);
    opt.Optimize();
    timer_.ProcedureFinish();

    ++iteration_;
    converge = opt.CheckConverge(tfs) || iteration_ == 100;
    cur_tf = opt.cur_tf3();
    tfs.push_back(cur_tf);
    if (HasOption(Options::kOneTime)) converge = true;
  }
  tfs_.insert(tfs_.end(), tfs.begin(), tfs.end());
  return cur_tf;
}

Eigen::Affine3d NDTMatcher::Align(const Eigen::Affine3d &guess) {
  timer_.Start();
  Eigen::Affine3d cur_tf = guess;
  if (HasOption(Options::kIterative)) {
    std::sort(cell_sizes_.begin(), cell_sizes_.end(), std::greater<>());
    for (auto cell_size : cell_sizes_) {
      cell_size_ = cell_size;
      cur_tf = AlignImpl(cur_tf);
    }
  } else {
    cur_tf = AlignImpl(cur_tf);
  }
  timer_.Finish();
  return cur_tf;
}

ICPMatcher ICPMatcher::GetBasic(std::unordered_set<Options> options) {
  return ICPMatcher(options);
}

ICPMatcher::ICPMatcher(std::unordered_set<Options> options)
    : Matcher(options) {}

Eigen::Affine3d ICPMatcher::Align(const Eigen::Affine3d &guess) {
  timer_.Start();
  auto cur_tf = guess;
  bool converge = false;

  pcl::KdTreeFLANN<pcl::PointXYZ> kd;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto &pt : tpts_) {
    pcl::PointXYZ p;
    p.x = pt(0), p.y = pt(1), p.z = pt(2);
    pc->push_back(p);
  }
  kd.setInputCloud(pc);

  std::vector<Eigen::Affine3d> tfs;
  tfs.push_back(cur_tf);

  while (!converge) {
    timer_.ProcedureStart(timer_.kBuild);
    auto next = TransformPoints(spts_, cur_tf);
    std::vector<Eigen::Vector3d> ps, qs;
    for (const auto &p : next) {
      pcl::PointXYZ query;
      query.x = p(0), query.y = p(1), query.z = p(2);
      std::vector<int> idx{0};
      std::vector<float> dist2{0};
      kd.nearestKSearch(query, 1, idx, dist2);
      auto q = tpts_[idx[0]];
      ps.push_back(p);
      qs.push_back(q);
    }

    if (!HasOption(Options::kNoReject)) {
      Orj orj(ps.size());
      // orj.RangeRejection(ps, qs, Rejection::kBoth, {1.});
      orj.RangeRejection(ps, qs, Rejection::kThreshold, {1.});
      orj.RetainIndices(ps, qs);
    }

    Options type = Options::kAnalytic;
    Optimizer opt(type);
    opt.set_cur_tf3(cur_tf);
    corres_ = ps.size();
    opt.BuildProblem(new ICPCost(ps, qs));
    timer_.ProcedureFinish();

    timer_.ProcedureStart(timer_.kOptimize);
    opt.Optimize();
    timer_.ProcedureFinish();

    ++iteration_;
    converge = opt.CheckConverge(tfs) || iteration_ == 100;
    cur_tf = opt.cur_tf3();
    tfs.push_back(cur_tf);
    if (HasOption(Options::kOneTime)) converge = true;
  }
  tfs_.insert(tfs_.end(), tfs.begin(), tfs.end());

  timer_.Finish();
  return cur_tf;
}

namespace {
void ComputeKDTreeAndNormals(const std::vector<Eigen::Vector3d> &points,
                             double radius,
                             bool use_omp,
                             pcl::search::Search<pcl::PointXYZ>::Ptr kd,
                             std::vector<Eigen::Vector3d> &normals) {
  normals.clear();
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto &pt : points) {
    pcl::PointXYZ p;
    p.x = pt(0), p.y = pt(1), p.z = pt(2);
    pc->push_back(p);
  }
  kd->setInputCloud(pc);
  pcl::PointCloud<pcl::Normal>::Ptr nm(new pcl::PointCloud<pcl::Normal>);
  pcl::Feature<pcl::PointXYZ, pcl::Normal>::Ptr est;
  if (use_omp)
    est = pcl::Feature<pcl::PointXYZ, pcl::Normal>::Ptr(
        new pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal>(8));
  else
    est = pcl::Feature<pcl::PointXYZ, pcl::Normal>::Ptr(
        new pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>);
  est->setInputCloud(pc);
  est->setSearchMethod(kd);
  est->setRadiusSearch(radius);
  est->compute(*nm);
  for (const auto &pt : *nm) {
    Eigen::Vector3d p(pt.normal_x, pt.normal_y, pt.normal_z);
    normals.push_back(p);
  }
}
}  // namespace

SICPMatcher SICPMatcher::GetBasic(std::unordered_set<Options> options,
                                  double radius) {
  return SICPMatcher(options, radius);
}

SICPMatcher::SICPMatcher(std::unordered_set<Options> options, double radius)
    : Matcher(options), radius_(radius) {}

Eigen::Affine3d SICPMatcher::Align(const Eigen::Affine3d &guess) {
  timer_.Start();
  auto cur_tf = guess;
  bool converge = false;

  timer_.ProcedureStart(timer_.kNormal);
  pcl::search::Search<pcl::PointXYZ>::Ptr spts_kd(
      new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::search::Search<pcl::PointXYZ>::Ptr tpts_kd(
      new pcl::search::KdTree<pcl::PointXYZ>);
  ComputeKDTreeAndNormals(spts_, radius_, HasOption(Options::kUseNormalOMP),
                          spts_kd, snms_);
  ComputeKDTreeAndNormals(tpts_, radius_, HasOption(Options::kUseNormalOMP),
                          tpts_kd, tnms_);
  timer_.ProcedureFinish();

  std::vector<Eigen::Affine3d> tfs;
  tfs.push_back(cur_tf);

  while (!converge) {
    timer_.ProcedureStart(timer_.kBuild);
    auto next = TransformPoints(spts_, cur_tf);
    auto nextnm = TransformNormals(snms_, cur_tf);
    std::vector<Eigen::Vector3d> ps, qs, nps, nqs;
    for (size_t i = 0; i < next.size(); ++i) {
      auto p = next[i];
      auto np = nextnm[i];
      if (!p.allFinite() || !np.allFinite()) continue;
      pcl::PointXYZ query;
      query.x = p(0), query.y = p(1), query.z = p(2);
      std::vector<int> idx{0};
      std::vector<float> dist2{0};
      tpts_kd->nearestKSearch(query, 1, idx, dist2);
      auto q = tpts_[idx[0]];
      auto nq = tnms_[idx[0]];
      if (!q.allFinite() || !nq.allFinite()) continue;
      if (np.dot(nq) < 0) np *= -1.;
      ps.push_back(p);
      nps.push_back(np);
      qs.push_back(q);
      nqs.push_back(nq);
    }

    if (!HasOption(Options::kNoReject)) {
      Orj orj(ps.size());
      // orj.RangeRejection(ps, qs, Rejection::kBoth, {1.});
      orj.RangeRejection(ps, qs, Rejection::kThreshold, {1.});
      orj.RetainIndices(ps, qs, nps, nqs);
    }
    Options type = Options::kAnalytic;
    Optimizer opt(type);
    opt.set_cur_tf3(cur_tf);
    corres_ = ps.size();
    opt.BuildProblem(new SICPCost(ps, nps, qs, nqs));
    timer_.ProcedureFinish();

    timer_.ProcedureStart(timer_.kOptimize);
    opt.Optimize();
    timer_.ProcedureFinish();

    ++iteration_;
    converge = opt.CheckConverge(tfs) || iteration_ == 100;
    cur_tf = opt.cur_tf3();
    tfs.push_back(cur_tf);
    if (HasOption(Options::kOneTime)) converge = true;
  }
  tfs_.insert(tfs_.end(), tfs.begin(), tfs.end());

  timer_.Finish();
  return cur_tf;
}
