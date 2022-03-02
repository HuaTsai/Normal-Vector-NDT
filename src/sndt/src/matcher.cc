/**
 * @file matcher.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Definition of Matching Algorithm
 * @version 0.1
 * @date 2021-07-29
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <ceres/ceres.h>
#include <common/eigen_utils.h>
#include <common/other_utils.h>
#include <normal2d/normal2d.h>
#include <sndt/cost_functors.h>
#include <sndt/matcher.h>
#include <sndt/outlier_reject.h>
#include <sndt/visuals.h>

template <typename T>
void RetainIndices(std::vector<T> &data, const std::vector<int> &ids) {
  int i = 0;
  for (auto id : ids) data[i++] = data[id];
  data.resize(ids.size());
}

Eigen::Affine2d ICPMatch(const std::vector<Eigen::Vector2d> &tpts,
                         const std::vector<Eigen::Vector2d> &spts,
                         ICPParameters &params,
                         const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);

  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_pts = TransformPoints(spts, opt.cur_tf());

    std::vector<Eigen::Vector2d> ps, qs;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx];
      if (!q.allFinite()) continue;
      ps.push_back(p);
      qs.push_back(q);
    }

    OutlierRejectionMaker orj(ps.size());
    if (params.reject) orj.RangeRejection(ps, qs, Rejection::kBoth, {1.5, 2});
    auto ids = orj.indices();
    for (auto id : ids)
      opt.AddResidualBlock(ICPCostFunctor::Create(ps[id], qs[id]));
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d SICPMatch(const std::vector<Eigen::Vector2d> &target_points,
                          const std::vector<Eigen::Vector2d> &source_points,
                          SICPParameters &params,
                          const Eigen::Affine2d &guess_tf) {
  params._usedtime.ProcedureStart(UsedTime::Procedure::kNormal);
  auto target_normals = ComputeNormals(target_points, params.radius);
  auto source_normals = ComputeNormals(source_points, params.radius);
  params._usedtime.ProcedureFinish();

  std::vector<Eigen::Vector2d> tpts, spts, tnms, snms;
  ExcludeInfinite(target_points, target_normals, tpts, tnms);
  ExcludeInfinite(source_points, source_normals, spts, snms);

  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_pts = TransformPoints(spts, opt.cur_tf());
    auto next_nms = TransformNormals(snms, opt.cur_tf());

    std::vector<Eigen::Vector2d> ps, qs, nps, nqs;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i], np = next_nms[i];
      if (!p.allFinite() || !np.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx], nq = tnms[idx];
      if (!q.allFinite() || !nq.allFinite()) continue;
      if (p.dot(np) < 0) np = -np;
      if (q.dot(nq) < 0) nq = -nq;
      if (np.dot(nq) < 0) np = -np;
      ps.push_back(p);
      qs.push_back(q);
      nps.push_back(np);
      nqs.push_back(nq);
    }

    OutlierRejectionMaker orj(ps.size());
    if (params.reject) orj.RangeRejection(ps, qs, Rejection::kBoth, {1.5, 2});
    auto ids = orj.indices();
    for (auto id : ids)
      opt.AddResidualBlock(
          SICPCostFunctor::Create(ps[id], nps[id], qs[id], nqs[id]));
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d D2DNDTMatch(const NDTMap &target_map,
                            const NDTMap &source_map,
                            D2DNDTParameters &params,
                            const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    GeneralOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_map = source_map.PseudoTransformCells(cur_tf);

    std::vector<Eigen::Vector2d> ups, uqs, nps, nqs;
    std::vector<Eigen::Matrix2d> cps, cqs;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      ups.push_back(cellp->GetPointMean());
      cps.push_back(cellp->GetPointCov());
      uqs.push_back(cellq->GetPointMean());
      cqs.push_back(cellq->GetPointCov());
      Eigen::Vector2d np = cellp->GetPointEvecs().col(0);
      Eigen::Vector2d nq = cellq->GetPointEvecs().col(0);
      if (ups.back().dot(np) < 0) np *= -1.;
      if (uqs.back().dot(nq) < 0) nq *= -1.;
      if (np.dot(nq) < 0) np = -np;
      nps.push_back(np);
      nqs.push_back(nq);
    }

    OutlierRejectionMaker orj(ups.size());
    if (params.reject)
      orj.RangeRejection(ups, uqs, Rejection::kBoth, {1.5, 2});
    auto ids = orj.indices();
    RetainIndices(ups, ids);
    RetainIndices(uqs, ids);
    RetainIndices(cps, ids);
    RetainIndices(cqs, ids);
    if (params.d2 == -1)
      opt.BuildProblem(D2DNDTCostFunctor2::Create(ups, cps, uqs, cqs,
                                                  params.cell_size, 0.2));
    else
      opt.BuildProblem(
          D2DNDTCostFunctor::Create(ups, cps, uqs, cqs, params.d2));
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d SNDTMatch2(const NDTMap &target_map,
                           const NDTMap &source_map,
                           D2DNDTParameters &params,
                           const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    GeneralOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf(), true);

    std::vector<Eigen::Vector2d> ups, nps, uqs, nqs;
    std::vector<Eigen::Matrix2d> cps, cqs;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      ups.push_back(cellp->GetPointMean());
      cps.push_back(cellp->GetPointCov());
      uqs.push_back(cellq->GetPointMean());
      cqs.push_back(cellq->GetPointCov());
      Eigen::Vector2d np = cellp->GetPointEvecs().col(0);
      Eigen::Vector2d nq = cellq->GetPointEvecs().col(0);
      if (ups.back().dot(np) < 0) np *= -1.;
      if (uqs.back().dot(nq) < 0) nq *= -1.;
      if (np.dot(nq) < 0) np = -np;
      nps.push_back(np);
      nqs.push_back(nq);
    }

    OutlierRejectionMaker orj(ups.size());
    if (params.reject)
      orj.RangeRejection(ups, uqs, Rejection::kBoth, {1.5, 2});
    auto ids = orj.indices();
    RetainIndices(ups, ids);
    RetainIndices(uqs, ids);
    RetainIndices(cps, ids);
    RetainIndices(cqs, ids);
    RetainIndices(nps, ids);
    RetainIndices(nqs, ids);
    opt.BuildProblem(
        SNDTCostFunctor2::Create(ups, cps, nps, uqs, cqs, nqs, params.d2));
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d IMatch(const std::vector<Eigen::Vector2d> &tpts,
                       const std::vector<Eigen::Vector2d> &spts,
                       ICPParameters &params,
                       const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);

  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    GeneralOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_pts = TransformPoints(spts, opt.cur_tf());

    std::vector<Eigen::Vector2d> ps, qs;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx];
      if (!q.allFinite()) continue;
      ps.push_back(p);
      qs.push_back(q);
    }

    OutlierRejectionMaker orj(ps.size());
    if (params.reject) orj.RangeRejection(ps, qs, Rejection::kBoth, {1.5, 2});
    auto ids = orj.indices();
    RetainIndices(ps, ids);
    RetainIndices(qs, ids);
    opt.BuildProblem(ICPCost::Create(ps, qs));
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d PMatch(const std::vector<Eigen::Vector2d> &target_points,
                       const std::vector<Eigen::Vector2d> &source_points,
                       SICPParameters &params,
                       const Eigen::Affine2d &guess_tf) {
  params._usedtime.ProcedureStart(UsedTime::Procedure::kNormal);
  auto target_normals = ComputeNormals(target_points, params.radius);
  auto source_normals = ComputeNormals(source_points, params.radius);
  params._usedtime.ProcedureFinish();

  std::vector<Eigen::Vector2d> tpts, spts, tnms, snms;
  ExcludeInfinite(target_points, target_normals, tpts, tnms);
  ExcludeInfinite(source_points, source_normals, spts, snms);

  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    GeneralOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_pts = TransformPoints(spts, opt.cur_tf());
    auto next_nms = TransformNormals(snms, opt.cur_tf());

    std::vector<Eigen::Vector2d> ps, qs, nps, nqs;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i], np = next_nms[i];
      if (!p.allFinite() || !np.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx], nq = tnms[idx];
      if (!q.allFinite() || !nq.allFinite()) continue;
      if (p.dot(np) < 0) np = -np;
      if (q.dot(nq) < 0) nq = -nq;
      if (np.dot(nq) < 0) np = -np;
      ps.push_back(p);
      qs.push_back(q);
      nps.push_back(np);
      nqs.push_back(nq);
    }

    OutlierRejectionMaker orj(ps.size());
    if (params.reject) orj.RangeRejection(ps, qs, Rejection::kBoth, {1.5, 2});
    auto ids = orj.indices();
    RetainIndices(ps, ids);
    RetainIndices(qs, ids);
    RetainIndices(nps, ids);
    RetainIndices(nqs, ids);
    opt.BuildProblem(SICPCost::Create(ps, nps, qs, nqs));
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d DMatch(const NDTMap &target_map,
                       const NDTMap &source_map,
                       D2DNDTParameters &params,
                       const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_map = source_map.PseudoTransformCells(cur_tf);

    std::vector<Eigen::Vector2d> ups, uqs, nps, nqs;
    std::vector<Eigen::Matrix2d> cps, cqs;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      ups.push_back(cellp->GetPointMean());
      cps.push_back(cellp->GetPointCov());
      uqs.push_back(cellq->GetPointMean());
      cqs.push_back(cellq->GetPointCov());
      Eigen::Vector2d np = cellp->GetPointEvecs().col(0);
      Eigen::Vector2d nq = cellq->GetPointEvecs().col(0);
      if (ups.back().dot(np) < 0) np *= -1.;
      if (uqs.back().dot(nq) < 0) nq *= -1.;
      if (np.dot(nq) < 0) np = -np;
      nps.push_back(np);
      nqs.push_back(nq);
    }

    OutlierRejectionMaker orj(ups.size());
    if (params.reject)
      orj.RangeRejection(ups, uqs, Rejection::kBoth, {1.5, 2});
    auto ids = orj.indices();
    for (auto id : ids)
      opt.AddResidualBlock(
          D2DNDTCost::Create(ups[id], cps[id], uqs[id], cqs[id], params.d2));
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d SMatch(const NDTMap &target_map,
                       const NDTMap &source_map,
                       D2DNDTParameters &params,
                       const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf(), true);

    std::vector<Eigen::Vector2d> ups, nps, uqs, nqs;
    std::vector<Eigen::Matrix2d> cps, cqs;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      ups.push_back(cellp->GetPointMean());
      cps.push_back(cellp->GetPointCov());
      uqs.push_back(cellq->GetPointMean());
      cqs.push_back(cellq->GetPointCov());
      Eigen::Vector2d np = cellp->GetPointEvecs().col(0);
      Eigen::Vector2d nq = cellq->GetPointEvecs().col(0);
      if (ups.back().dot(np) < 0) np *= -1.;
      if (uqs.back().dot(nq) < 0) nq *= -1.;
      if (np.dot(nq) < 0) np = -np;
      nps.push_back(np);
      nqs.push_back(nq);
    }

    OutlierRejectionMaker orj(ups.size());
    if (params.reject)
      orj.RangeRejection(ups, uqs, Rejection::kBoth, {1.5, 2});
    auto ids = orj.indices();
    for (auto id : ids)
      opt.AddResidualBlock(SNDTCost::Create(ups[id], cps[id], nps[id], uqs[id],
                                            cqs[id], nqs[id], params.d2));
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

#include "matcher_arc.cc"
