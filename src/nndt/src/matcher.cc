#include <nndt/costs.h>
#include <nndt/matcher.h>
#include <nndt/opt.h>
#include <nndt/orj.h>

template <typename T>
void RetainIndices(std::vector<T> &data, const std::vector<int> &ids) {
  int i = 0;
  for (auto id : ids) data[i++] = data[id];
  data.resize(ids.size());
}

NDTMatcher::NDTMatcher(MatchType type, double cell_size, double d2)
    : type_(type), cell_size_(cell_size), d2_(d2), iteration_(0) {
  timer_.Start();
}

void NDTMatcher::SetTarget(const std::vector<Eigen::Vector3d> &points) {
  timer_.ProcedureStart(timer_.kNDT);
  tmap_ = std::make_shared<NMap>(cell_size_);
  tmap_->LoadPoints(points);
  timer_.ProcedureFinish();
}

void NDTMatcher::SetSource(const std::vector<Eigen::Vector3d> &points) {
  timer_.ProcedureStart(timer_.kNDT);
  smap_ = std::make_shared<NMap>(cell_size_);
  smap_->LoadPoints(points);
  timer_.ProcedureFinish();
}

Eigen::Affine3d NDTMatcher::Align(const Eigen::Affine3d &guess) {
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
    }

    Orj orj(ups.size());
    // orj.RangeRejection(ups, uqs, Orj::Rejection::kBoth, {1.5, 2});
    // orj.RangeRejection(ups, uqs, Orj::Rejection::kStatistic, {1.5});
    orj.RangeRejection(ups, uqs, Orj::Rejection::kThreshold, {2});
    auto ids = orj.indices();
    RetainIndices(ups, ids);
    RetainIndices(uqs, ids);
    RetainIndices(cps, ids);
    RetainIndices(cqs, ids);
    RetainIndices(nps, ids);
    RetainIndices(nqs, ids);

    Optimizer::OptType type = Optimizer::OptType::kLS;
    if (type_ == kNDTLS || type_ == kNNDTLS)
      type = Optimizer::OptType::kLS;
    else if (type_ == kNDTTR || type_ == kNNDTTR)
      type = Optimizer::OptType::kTR;
    Optimizer opt(type);
    opt.set_cur_tf(cur_tf);
    corres_ = ups.size();
    if (type_ == kNDTLS) {
      opt.BuildProblem(NDTCost::Create(ups, cps, uqs, cqs, d2_));
    } else if (type_ == kNNDTLS) {
      opt.BuildProblem(NNDTCost::Create(ups, cps, nps, uqs, cqs, nqs, d2_));
    } else if (type_ == kNDTTR) {
      for (size_t i = 0; i < ups.size(); ++i)
        opt.AddResidualBlock(
            NDTCostTR::Create(ups[i], cps[i], uqs[i], cqs[i], d2_));
    } else if (type_ == kNNDTTR) {
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
  timer_.Finish();
  return cur_tf;
}