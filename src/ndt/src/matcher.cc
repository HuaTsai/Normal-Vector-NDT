#include <ndt/costs.h>
#include <ndt/matcher.h>
#include <ndt/opt.h>
#include <ndt/orj.h>

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
    : options_(options),
      cell_sizes_(cell_sizes),
      cell_size_(cell_size),
      d2_(d2),
      intrinsic_(intrinsic),
      iteration_(0),
      corres_(0) {
  if ((HasOption(Options::kNDT) && HasOption(Options::kNormalNDT)) ||
      (HasOption(Options::k1to1) && HasOption(Options::k1ton))) {
    std::cerr << __FUNCTION__ << ": invalid options\n";
    std::exit(1);
  }
}

bool NDTMatcher::HasOption(Options option) { return options_.count(option); }

void NDTMatcher::SetTarget(const std::vector<Eigen::Vector3d> &points) {
  tpts_ = points;
}

void NDTMatcher::SetSource(const std::vector<Eigen::Vector3d> &points) {
  spts_ = points;
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
      orj.RangeRejection(ups, uqs, Rejection::kBoth, {cell_size_});
      orj.AngleRejection(nps, nqs, Rejection::kThreshold, {1});
      orj.RetainIndices(ups, uqs, cps, cqs, nps, nqs);
    }

    Options type = Options::kOptimizer3D;
    if (HasOption(Options::kLBFGSPP))
      type = Options::kLBFGSPP;
    Optimizer opt(type);
    opt.set_cur_tf3(cur_tf);
    corres_ = ups.size();
    if (HasOption(Options::kNDT)) {
      if (HasOption(Options::kLBFGSPP))
        opt.BuildProblem(new CostObj(ups, cps, uqs, cqs, d2_));
      else
        opt.BuildProblem(NDTCost::Create(ups, cps, uqs, cqs, d2_));
    } else if (HasOption(Options::kNormalNDT)) {
      opt.BuildProblem(NNDTCost::Create(ups, cps, nps, uqs, cqs, nqs, d2_));
    }
    timer_.ProcedureFinish();

    timer_.ProcedureStart(timer_.kOptimize);
    opt.Optimize();
    timer_.ProcedureFinish();

    ++iteration_;
    converge = opt.CheckConverge(tfs) || iteration_ == 100;
    cur_tf = opt.cur_tf3();
    tfs.push_back(cur_tf);
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
