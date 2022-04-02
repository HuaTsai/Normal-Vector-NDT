#include <ndt/costs.h>
#include <ndt/matcher.h>
#include <ndt/opt.h>
#include <ndt/orj.h>

NDTMatcher::NDTMatcher(std::unordered_set<Options> options,
                       double cell_size,
                       double d2)
    : options_(options),
      cell_size_(cell_size),
      d2_(d2),
      iteration_(0),
      corres_(0) {
  if ((HasOption(Options::kLineSearch) && HasOption(Options::kTrustRegion)) ||
      (HasOption(Options::kNDT) && HasOption(Options::kNormalNDT)) ||
      (HasOption(Options::k1to1) && HasOption(Options::k1ton))) {
    std::cerr << __FUNCTION__ << ": invalid options\n";
    std::exit(1);
  }
}

NDTMatcher::NDTMatcher(std::unordered_set<Options> options,
                       std::vector<double> cell_sizes,
                       double d2)
    : options_(options),
      cell_sizes_(cell_sizes),
      d2_(d2),
      iteration_(0),
      corres_(0) {
  if ((HasOption(Options::kLineSearch) && HasOption(Options::kTrustRegion)) ||
      (HasOption(Options::kNDT) && HasOption(Options::kNormalNDT)) ||
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
  timer_.ProcedureStart(timer_.kNDT);
  tmap_ = std::make_shared<NMap>(cell_size_);
  tmap_->LoadPoints(tpts_);
  timer_.ProcedureFinish();

  timer_.ProcedureStart(timer_.kNDT);
  smap_ = std::make_shared<NMap>(cell_size_);
  smap_->LoadPoints(spts_);
  timer_.ProcedureFinish();

  auto cur_tf = guess;
  bool converge = false;

  std::vector<Eigen::Affine3d> tfs;
  tfs.push_back(cur_tf);

  bool ls = HasOption(Options::kLineSearch);
  bool tr = HasOption(Options::kTrustRegion);
  bool ndt = HasOption(Options::kNDT);
  bool nndt = HasOption(Options::kNormalNDT);
  bool to1 = HasOption(Options::k1to1);
  bool ton = HasOption(Options::k1ton);
  while (!converge) {
    timer_.ProcedureStart(timer_.kBuild);
    auto next = smap_->TransformCells(cur_tf);
    std::vector<Eigen::Vector3d> ups, uqs;
    std::vector<Eigen::Matrix3d> cps, cqs;
    std::vector<Eigen::Vector3d> nps, nqs;
    for (const auto &cellp : next) {
      if (!cellp.GetHasGaussian()) continue;
      if (to1) {
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
      } else if (ton) {
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

    Orj orj(ups.size());
    // orj.RangeRejection(ups, uqs, Orj::Rejection::kBoth, {1.5, 2});
    // orj.RangeRejection(ups, uqs, Orj::Rejection::kStatistic, {1.5});
    // orj.RangeRejection(ups, uqs, Orj::Rejection::kThreshold, {2});
    orj.RetainIndices(ups, uqs, cps, cqs, nps, nqs);

    Optimizer opt(ls ? Optimizer::OptType::kLS : Optimizer::OptType::kTR);
    opt.set_cur_tf(cur_tf);
    corres_ = ups.size();
    if (ls && ndt) {
      opt.BuildProblem(NDTCost::Create(ups, cps, uqs, cqs, d2_));
    } else if (ls && nndt) {
      opt.BuildProblem(NNDTCost::Create(ups, cps, nps, uqs, cqs, nqs, d2_));
    } else if (tr && ndt) {
      for (size_t i = 0; i < ups.size(); ++i)
        opt.AddResidualBlock(
            NDTCostTR::Create(ups[i], cps[i], uqs[i], cqs[i], d2_));
    } else if (tr && nndt) {
      for (size_t i = 0; i < ups.size(); ++i)
        opt.AddResidualBlock(NNDTCostTR::Create(ups[i], cps[i], nps[i], uqs[i],
                                                cqs[i], nqs[i], d2_));
    }
    timer_.ProcedureFinish();

    timer_.ProcedureStart(timer_.kOptimize);
    opt.Optimize();
    timer_.ProcedureFinish();

    ++iteration_;
    converge = opt.CheckConverge(tfs) || iteration_ == 100;
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
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
