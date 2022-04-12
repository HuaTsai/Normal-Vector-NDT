#include <ndt/costs.h>
#include <ndt/matcher2d.h>
#include <ndt/opt.h>
#include <ndt/orj.h>

NDTMatcher2D NDTMatcher2D::GetIter(std::unordered_set<Options> options,
                                   std::vector<double> cell_sizes,
                                   double d2) {
  options.insert(Options::kIterative);
  return NDTMatcher2D(options, cell_sizes, 0, d2, 0.005);
}

NDTMatcher2D NDTMatcher2D::GetBasic(std::unordered_set<Options> options,
                                    double cell_size,
                                    double d2) {
  if (options.count(Options::kIterative)) {
    std::cerr << __FUNCTION__ << ": invalid options\n";
    std::exit(1);
  }
  return NDTMatcher2D(options, {}, cell_size, d2, 0.005);
}

NDTMatcher2D::NDTMatcher2D(std::unordered_set<Options> options,
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

bool NDTMatcher2D::HasOption(Options option) { return options_.count(option); }

void NDTMatcher2D::SetTarget(const std::vector<Eigen::Vector2d> &points) {
  tpts_ = points;
}

void NDTMatcher2D::SetSource(const std::vector<Eigen::Vector2d> &points) {
  spts_ = points;
}

Eigen::Affine2d NDTMatcher2D::AlignImpl(const Eigen::Affine2d &guess) {
  // the value here may be crucial!
  Eigen::Matrix2d pcov = Eigen::Matrix2d::Identity() * intrinsic_;
  timer_.ProcedureStart(timer_.kNDT);
  tmap_ = std::make_shared<NMap2D>(cell_size_);
  if (HasOption(Options::kPointCov))
    tmap_->LoadPointsWithCovariances(tpts_, pcov);
  else
    tmap_->LoadPoints(tpts_);
  timer_.ProcedureFinish();

  timer_.ProcedureStart(timer_.kNDT);
  smap_ = std::make_shared<NMap2D>(cell_size_);
  if (HasOption(Options::kPointCov))
    smap_->LoadPointsWithCovariances(spts_, pcov);
  else
    smap_->LoadPoints(spts_);
  timer_.ProcedureFinish();

  auto cur_tf = guess;
  bool converge = false;

  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);

  while (!converge) {
    timer_.ProcedureStart(timer_.kBuild);
    auto next = smap_->TransformCells(cur_tf);
    std::vector<Eigen::Vector2d> ups, uqs;
    std::vector<Eigen::Matrix2d> cps, cqs;
    std::vector<Eigen::Vector2d> nps, nqs;
    for (const auto &cellp : next) {
      if (!cellp.GetHasGaussian()) continue;
      if (HasOption(Options::k1to1)) {
        auto cellq = tmap_->SearchNearestCell(cellp.GetMean());
        if (!cellq.GetHasGaussian()) continue;
        Eigen::Vector2d np = cellp.GetNormal();
        Eigen::Vector2d nq = cellq.GetNormal();
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
          const Cell2D &cellq = c.get();
          if (!cellq.GetHasGaussian()) continue;
          Eigen::Vector2d np = cellp.GetNormal();
          Eigen::Vector2d nq = cellq.GetNormal();
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
      Orj2D orj(ups.size());
      orj.RangeRejection(ups, uqs, Rejection::kBoth, {cell_size_});
      orj.AngleRejection(nps, nqs, Rejection::kThreshold, {1});
      orj.RetainIndices(ups, uqs, cps, cqs, nps, nqs);
    }

    Optimizer opt(Options::kOptimizer2D);
    opt.set_cur_tf2(cur_tf);
    corres_ = ups.size();
    if (HasOption(Options::kNDT))
      opt.BuildProblem(NDTCost2D::Create(ups, cps, uqs, cqs, d2_));
    else if (HasOption(Options::kNormalNDT))
      opt.BuildProblem(NNDTCost2D::Create(ups, cps, nps, uqs, cqs, nqs, d2_));
    timer_.ProcedureFinish();

    timer_.ProcedureStart(timer_.kOptimize);
    opt.Optimize();
    timer_.ProcedureFinish();

    ++iteration_;
    converge = opt.CheckConverge(tfs) || iteration_ == 100;
    cur_tf = opt.cur_tf2();
    tfs.push_back(cur_tf);
  }
  tfs_.insert(tfs_.end(), tfs.begin(), tfs.end());
  return cur_tf;
}

Eigen::Affine2d NDTMatcher2D::Align(const Eigen::Affine2d &guess) {
  timer_.Start();
  Eigen::Affine2d cur_tf = guess;
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
