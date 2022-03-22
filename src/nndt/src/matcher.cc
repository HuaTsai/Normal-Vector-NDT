#include <nndt/costs.h>
#include <nndt/matcher.h>
#include <nndt/opt.h>

NDTMatcher::NDTMatcher(MatchType type, double cell_size)
    : type_(type), cell_size_(cell_size), iteration_(0) {
  timer_.Start();
}

void NDTMatcher::SetTarget(const std::vector<Eigen::Vector3d> &points) {
  timer_.ProcedureStart(timer_.kNDT);
  tmap = std::make_shared<NMap>(cell_size_);
  tmap->LoadPoints(points);
  timer_.ProcedureFinish();
}

void NDTMatcher::SetSource(const std::vector<Eigen::Vector3d> &points) {
  timer_.ProcedureStart(timer_.kNDT);
  smap = std::make_shared<NMap>(cell_size_);
  smap->LoadPoints(points);
  timer_.ProcedureFinish();
}

Eigen::Affine3d NDTMatcher::Align(const Eigen::Affine3d &guess) {
  auto cur_tf = guess;
  bool converge = false;
  while (!converge) {
    timer_.ProcedureStart(timer_.kBuild);
    auto next = smap->TransformCells(cur_tf);
    std::vector<Eigen::Vector3d> ups, uqs;
    std::vector<Eigen::Matrix3d> cps, cqs;
    std::vector<Eigen::Vector3d> nps, nqs;
    for (const auto &cellp : next) {
      if (!cellp.GetHasGaussian()) continue;
      auto cellq = tmap->SearchNearestCell(cellp.GetMean());
      if (!cellq.GetHasGaussian()) continue;
      Eigen::Vector3d np, nq;
      // FIXME: should kNDT here?
      if (type_ == kNNDT || type_ == kNDT) {
        if (!cellp.Normal(np)) continue;
        if (!cellq.Normal(nq)) continue;
        if (np.dot(nq) < 0) np = -np;
        nps.push_back(np);
        nqs.push_back(nq);
      }
      ups.push_back(cellp.GetMean());
      cps.push_back(cellp.GetCov());
      uqs.push_back(cellq.GetMean());
      cqs.push_back(cellq.GetCov());
    }
    Optimizer opt;
    opt.set_cur_tf(cur_tf);
    if (type_ == kNDT)
      opt.BuildProblem(NDTCost::Create(ups, cps, uqs, cqs, 0.05));
    else
      opt.BuildProblem(NNDTCost::Create(ups, cps, nps, uqs, cqs, nqs, 0.05));
    timer_.ProcedureFinish();

    timer_.ProcedureStart(timer_.kOptimize);
    opt.Optimize();
    timer_.ProcedureFinish();

    ++iteration_;
    converge = opt.CheckConverge() || iteration_ == 100;
    cur_tf = opt.cur_tf();
  }
  timer_.Finish();
  return cur_tf;
}