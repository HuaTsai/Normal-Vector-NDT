Eigen::Affine2d Pt2plICPMatch(const std::vector<Eigen::Vector2d> &target_points,
                              const std::vector<Eigen::Vector2d> &spts,
                              Pt2plICPParameters &params,
                              const Eigen::Affine2d &guess_tf) {
  params._usedtime.ProcedureStart(UsedTime::Procedure::kNormal);
  auto target_normals = ComputeNormals(target_points, params.radius);
  params._usedtime.ProcedureFinish();
  std::vector<Eigen::Vector2d> tpts, tnms;
  ExcludeInfinite(target_points, target_normals, tpts, tnms);

  auto kd = MakeKDTree(tpts);

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_pts = TransformPoints(spts, opt.cur_tf());

    RangeOutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      Eigen::Vector2d p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      Eigen::Vector2d q = tpts[idx], nq = tnms[idx];
      if (!q.allFinite() || !nq.allFinite()) continue;
      corres.push_back({i, idx});
      orj.AddCorrespondence(p, q);
      residuals.push_back(Pt2plICPCostFunctor::Create(p, q, nq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }
    params._usedtime.ProcedureFinish();

    opt.Optimize(params);
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d P2DNDTMDMatch(const NDTMap &target_map,
                              const std::vector<Eigen::Vector2d> &spts,
                              P2DNDTParameters &params,
                              const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    LeastSquareOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_pts = TransformPoints(spts, opt.cur_tf());

    RangeOutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      auto p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      corres.push_back({i, target_map.GetCellIndex(cellq)});
      orj.AddCorrespondence(p, cellq->GetPointMean());
      auto q = cellq->GetPointMean();
      auto cq = cellq->GetPointCov();
      residuals.push_back(P2DNDTMDCostFunctor::Create(p, q, cq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }
    params._usedtime.ProcedureFinish();

    params._usedtime.ProcedureStart(UsedTime::Procedure::kOptimize);
    opt.Optimize(params);
    params._usedtime.ProcedureFinish();
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d P2DNDTMatch(const NDTMap &target_map,
                            const std::vector<Eigen::Vector2d> &spts,
                            P2DNDTParameters &params,
                            const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPointsWithGaussianCell());

  auto cur_tf = guess_tf;
  std::vector<Eigen::Affine2d> tfs;
  tfs.push_back(cur_tf);
  while (params._converge == Converge::kNotConverge) {
    params._usedtime.ProcedureStart(UsedTime::Procedure::kBuild);
    GeneralOptimize opt;
    opt.set_cur_tf(cur_tf);
    auto next_pts = TransformPoints(spts, opt.cur_tf());

    RangeOutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<Eigen::Vector2d> ps, uqs;
    std::vector<Eigen::Matrix2d> cqs;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_pts.size(); ++i) {
      auto p = next_pts[i];
      if (!p.allFinite()) continue;
      auto idx = FindNearestNeighborIndex(p, kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      corres.push_back({i, target_map.GetCellIndex(cellq)});
      ps.push_back(p);
      uqs.push_back(cellq->GetPointMean());
      cqs.push_back(cellq->GetPointCov());
      orj.AddCorrespondence(p, cellq->GetPointMean());
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    std::vector<Eigen::Vector2d> ps2, uqs2;
    std::vector<Eigen::Matrix2d> cqs2;
    for (size_t i = 0; i < indices.size(); ++i) {
      ps2.push_back(ps[i]);
      uqs2.push_back(uqs[i]);
      cqs2.push_back(cqs[i]);
      params._corres.back().push_back(corres[i]);
    }

    opt.BuildProblem(P2DNDTCostFunctor::Create(params.d2, ps2, uqs2, cqs2));
    params._usedtime.ProcedureFinish();

    params._usedtime.ProcedureStart(UsedTime::Procedure::kOptimize);
    opt.Optimize(params);
    params._usedtime.ProcedureFinish();
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d D2DNDTMDMatch(const NDTMap &target_map,
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
    auto next_map = source_map.PseudoTransformCells(opt.cur_tf());

    RangeOutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      corres.push_back({i, target_map.GetCellIndex(cellq)});
      auto up = cellp->GetPointMean();
      auto cp = cellp->GetPointCov();
      auto uq = cellq->GetPointMean();
      auto cq = cellq->GetPointCov();
      orj.AddCorrespondence(up, uq);
      residuals.push_back(D2DNDTMDCostFunctor::Create(up, cp, uq, cq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }
    params._usedtime.ProcedureFinish();

    params._usedtime.ProcedureStart(UsedTime::Procedure::kOptimize);
    opt.Optimize(params);
    params._usedtime.ProcedureFinish();
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d SNDTMDMatch(const SNDTMap &target_map,
                            const SNDTMap &source_map,
                            SNDTParameters &params,
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

    RangeOutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      corres.push_back({i, target_map.GetCellIndex(cellq)});
      auto up = cellp->GetPointMean();
      auto cp = cellp->GetPointCov();
      auto unp = cellp->GetNormalMean();
      auto cnp = cellp->GetNormalCov();
      auto uq = cellq->GetPointMean();
      auto cq = cellq->GetPointCov();
      auto unq = cellq->GetNormalMean();
      auto cnq = cellq->GetNormalCov();
      orj.AddCorrespondence(up, uq);
      residuals.push_back(
          SNDTMDCostFunctor::Create(up, cp, unp, cnp, uq, cq, unq, cnq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }
    params._usedtime.ProcedureFinish();

    params._usedtime.ProcedureStart(UsedTime::Procedure::kOptimize);
    opt.Optimize(params);
    params._usedtime.ProcedureFinish();
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d SNDTMatch(const SNDTMap &target_map,
                          const SNDTMap &source_map,
                          SNDTParameters &params,
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

    RangeOutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<Eigen::Vector2d> ups, unps, uqs, unqs;
    std::vector<Eigen::Matrix2d> cps, cnps, cqs, cnqs;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      corres.push_back({i, target_map.GetCellIndex(cellq)});
      ups.push_back(cellp->GetPointMean());
      cps.push_back(cellp->GetPointCov());
      unps.push_back(cellp->GetNormalMean());
      cnps.push_back(cellp->GetNormalCov());
      uqs.push_back(cellq->GetPointMean());
      cqs.push_back(cellq->GetPointCov());
      unqs.push_back(cellq->GetNormalMean());
      cnqs.push_back(cellq->GetNormalCov());
      orj.AddCorrespondence(cellp->GetPointMean(), cellq->GetPointMean());
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    // XXX: we should change in place...
    std::vector<Eigen::Vector2d> ups2, unps2, uqs2, unqs2;
    std::vector<Eigen::Matrix2d> cps2, cnps2, cqs2, cnqs2;
    for (size_t i = 0; i < indices.size(); ++i) {
      ups2.push_back(ups[i]);
      cps2.push_back(cps[i]);
      unps2.push_back(unps[i]);
      cnps2.push_back(cnps[i]);
      uqs2.push_back(uqs[i]);
      cqs2.push_back(cqs[i]);
      unqs2.push_back(unqs[i]);
      cnqs2.push_back(cnqs[i]);
      params._corres.back().push_back(corres[i]);
    }

    opt.BuildProblem(SNDTCostFunctor::Create(params.d2, ups2, cps2, unps2,
                                             cnps2, uqs2, cqs2, unqs2, cnqs2));
    params._usedtime.ProcedureFinish();

    params._usedtime.ProcedureStart(UsedTime::Procedure::kOptimize);
    opt.Optimize(params);
    params._usedtime.ProcedureFinish();
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}

Eigen::Affine2d SNDTMDMatch2(const NDTMap &target_map,
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

    RangeOutlierRejection orj;
    orj.set_reject(params.reject);
    std::vector<ceres::CostFunction *> residuals;
    std::vector<std::pair<int, int>> corres;
    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      auto idx = FindNearestNeighborIndex(cellp->GetPointMean(), kd);
      if (idx == -1) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx).x, kd.getInputCloud()->at(idx).y));
      if (!cellq || !cellq->HasGaussian()) continue;
      corres.push_back({i, target_map.GetCellIndex(cellq)});
      auto up = cellp->GetPointMean();
      auto cp = cellp->GetPointCov();
      Eigen::Vector2d unp = cellp->GetPointEvecs().col(0);
      auto uq = cellq->GetPointMean();
      auto cq = cellq->GetPointCov();
      Eigen::Vector2d unq = cellq->GetPointEvecs().col(0);
      orj.AddCorrespondence(up, uq);
      residuals.push_back(SNDTMDCostFunctor2::Create(up, cp, unp, uq, cq, unq));
    }
    params._corres.push_back({});
    auto indices = orj.GetIndices();
    for (size_t i = 0; i < indices.size(); ++i) {
      opt.AddResidualBlock(residuals[indices[i]]);
      params._corres.back().push_back(corres[i]);
    }
    params._usedtime.ProcedureFinish();

    params._usedtime.ProcedureStart(UsedTime::Procedure::kOptimize);
    opt.Optimize(params);
    params._usedtime.ProcedureFinish();
    opt.CheckConverge(params, tfs);
    cur_tf = opt.cur_tf();
    tfs.push_back(cur_tf);
  }
  params._usedtime.Finish();
  return cur_tf;
}
