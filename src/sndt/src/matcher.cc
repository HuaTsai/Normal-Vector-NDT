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
#include <sndt/matcher.h>
#include <sndt/pcl_utils.h>
#include <sndt/visuals.h>

template <typename T>
Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw) {
  T cos_yaw = ceres::cos(yaw);
  T sin_yaw = ceres::sin(yaw);
  Eigen::Matrix<T, 2, 2> ret;
  ret << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  return ret;
}

class AngleLocalParameterization {
 public:
  template <typename T>
  bool operator()(const T *t, const T *dt, T *t_add_dt) const {
    T pi(M_PI), twopi(2. * M_PI), sum(*t + *dt);
    *t_add_dt = sum - twopi * ceres::floor((sum + pi) / twopi);
    return true;
  }

  static ceres::LocalParameterization *Create() {
    return new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>;
  }
};

struct SICPCostFunctor {
  const Eigen::Vector2d p, np, q, nq;
  SICPCostFunctor(const Eigen::Vector2d &p_, const Eigen::Vector2d &np_,
                  const Eigen::Vector2d &q_, const Eigen::Vector2d &nq_)
      : p(p_), np(np_), q(q_), nq(nq_) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> p2 = R * p.cast<T>() + t;
    Eigen::Matrix<T, 2, 1> np2 = R * np.cast<T>();

    Eigen::Matrix<T, 2, 1> q2 = q.cast<T>();
    Eigen::Matrix<T, 2, 1> nq2 = nq.cast<T>();

    *e = abs((p2 - q2).dot(np2 + nq2));
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector2d &p,
                                     const Eigen::Vector2d &np,
                                     const Eigen::Vector2d &q,
                                     const Eigen::Vector2d &nq) {
    return new ceres::AutoDiffCostFunction<SICPCostFunctor, 1, 1, 1, 1>(
        new SICPCostFunctor(p, np, q, nq));
  }
};

Eigen::Affine2d SICPMatch(
    const std::vector<Eigen::Vector2d> &target_points,
    const std::vector<Eigen::Vector2d> &source_points,
    const SICPParameters &params,
    const Eigen::Affine2d &guess_tf) {
  auto tpts = ExcludeNaNInf(target_points);
  auto spts = ExcludeNaNInf(source_points);
  auto kd = MakeKDTree(tpts);
  auto tnms = ComputeNormals(tpts, params.radius);
  auto snms = ComputeNormals(spts, params.radius);
  bool converge = false;
  int iteration = 0;
  auto cur_tf = guess_tf;
  int n = source_points.size();
  int min_ctr = 0;
  double min_cost = std::numeric_limits<double>::max();

  while (!converge) {
    double x = 0, y = 0, t = 0;
    std::vector<Eigen::Vector2d> next_pts(n);
    std::vector<Eigen::Vector2d> next_nms(n);
    std::transform(spts.begin(), spts.end(), next_pts.begin(), [&cur_tf](auto p) { return cur_tf * p; });
    std::transform(snms.begin(), snms.end(), next_nms.begin(), [&cur_tf](auto p) { return cur_tf.rotation() * p; });

    ceres::Problem problem;
    for (int i = 0; i < n; ++i) {
      auto p = next_pts[i];
      auto np = next_nms[i];
      if (!p.allFinite() || !np.allFinite()) continue;
      pcl::PointXY pt;
      pt.x = p(0), pt.y = p(1);
      std::vector<int> idx{0};
      std::vector<float> dist2{0};
      int found = kd.nearestKSearch(pt, 1, idx, dist2);
      if (!found) continue;

      auto q = tpts[idx[0]];
      auto nq = tnms[idx[0]];
      if (!nq.allFinite()) continue;

      if (params.huber != 0) {
        ceres::LossFunction *loss(new ceres::HuberLoss(params.huber));
        problem.AddResidualBlock(SICPCostFunctor::Create(p, np, q, nq), loss,
                                 &x, &y, &t);
      } else {
        problem.AddResidualBlock(SICPCostFunctor::Create(p, np, q, nq), nullptr,
                                 &x, &y, &t);
      }
    }

    problem.SetParameterization(&t, AngleLocalParameterization::Create());

    ceres::Solver::Options options;
    if (params.verbose) options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = params.max_iterations;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cur_tf = Eigen::Rotation2Dd(t) * Eigen::Translation2d(x, y) * cur_tf;

    if (params.verbose) {
      std::cout << "Iteration " << iteration << ": " << std::endl;
      std::cout << summary.BriefReport() << std::endl;
    }

    if (min_cost < summary.final_cost) {
      ++min_ctr;
    } else {
      min_ctr = 0;
      min_cost = summary.final_cost;
    }

    converge = (min_ctr == 5) ||
               (Eigen::Vector2d(x, y).norm() < params.threshold) ||
               (iteration > params.max_iterations);

    // converge = true;
    ++iteration;
  }
  return cur_tf;
}

struct NDTD2DCostFunctor {
  const NDTCell *p, *q;
  NDTD2DCostFunctor(const NDTCell *p_, const NDTCell *q_) : p(p_), q(q_) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> up = R * p->GetPointMean().cast<T>() + t;
    Eigen::Matrix<T, 2, 2> cp = R * p->GetPointCov().cast<T>() * R.transpose();

    Eigen::Matrix<T, 2, 1> uq = q->GetPointMean().cast<T>();
    Eigen::Matrix<T, 2, 2> cq = q->GetPointCov().cast<T>();

    Eigen::Matrix<T, 2, 1> m = up - uq;
    Eigen::Matrix<T, 2, 2> c = cp + cq;

    *e = sqrt((m.transpose() * c.inverse() * m)[0]);
    return true;
  }

  static ceres::CostFunction *Create(const NDTCell *cellp,
                                     const NDTCell *cellq) {
    return new ceres::AutoDiffCostFunction<NDTD2DCostFunctor, 1, 1, 1, 1>(
        new NDTD2DCostFunctor(cellp, cellq));
  }
};

Eigen::Affine2d NDTD2DMatch(
    const NDTMap &target_map, const NDTMap &source_map,
    const NDTD2DParameters &params,
    const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPoints());
  bool converge = false;
  int iteration = 0;
  auto cur_tf = guess_tf;
  int min_ctr = 0;
  double min_cost = std::numeric_limits<double>::max();

  while (!converge) {
    double x = 0, y = 0, t = 0;
    auto next_map = source_map.PseudoTransformCells(cur_tf);
    ceres::Problem problem;

    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      pcl::PointXY pt;
      pt.x = cellp->GetPointMean()(0), pt.y = cellp->GetPointMean()(1);
      std::vector<int> idx{0};
      std::vector<float> dist2{0};
      int found = kd.nearestKSearch(pt, 1, idx, dist2);
      if (!found) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx[0]).x, kd.getInputCloud()->at(idx[0]).y));
      if (!cellq || !cellq->HasGaussian()) continue;

      if (params.huber != 0) {
        ceres::LossFunction *loss(new ceres::HuberLoss(params.huber));
        problem.AddResidualBlock(NDTD2DCostFunctor::Create(cellp.get(), cellq),
                                 loss, &x, &y, &t);
      } else {
        problem.AddResidualBlock(NDTD2DCostFunctor::Create(cellp.get(), cellq),
                                 nullptr, &x, &y, &t);
      }
    }

    problem.SetParameterization(&t, AngleLocalParameterization::Create());

    ceres::Solver::Options options;
    if (params.verbose) options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = params.max_iterations;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cur_tf = Eigen::Rotation2Dd(t) * Eigen::Translation2d(x, y) * cur_tf;

    if (params.verbose) {
      std::cout << "Iteration " << iteration << ": " << std::endl;
      std::cout << summary.BriefReport() << std::endl;
    }

    if (min_cost < summary.final_cost) {
      ++min_ctr;
    } else {
      min_ctr = 0;
      min_cost = summary.final_cost;
    }
    converge = (min_ctr == 5) ||
               (Eigen::Vector2d(x, y).norm() < params.threshold) ||
               (iteration > params.max_iterations);

    // converge = true;
    ++iteration;
  }
  return cur_tf;
}

struct SNDTCostFunctor {
  const SNDTCell *p, *q;
  SNDTCostFunctor(const SNDTCell *p_, const SNDTCell *q_) : p(p_), q(q_) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw,
                  T *e) const {
    Eigen::Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Eigen::Matrix<T, 2, 1> t(*x, *y);

    Eigen::Matrix<T, 2, 1> up = R * p->GetPointMean().cast<T>() + t;
    Eigen::Matrix<T, 2, 1> unp = R * p->GetNormalMean().cast<T>();
    Eigen::Matrix<T, 2, 2> cp = R * p->GetPointCov().cast<T>() * R.transpose();
    Eigen::Matrix<T, 2, 2> cnp = R * p->GetNormalCov().cast<T>() * R.transpose();

    Eigen::Matrix<T, 2, 1> uq = q->GetPointMean().cast<T>();
    Eigen::Matrix<T, 2, 1> unq = q->GetNormalMean().cast<T>();
    Eigen::Matrix<T, 2, 2> cq = q->GetPointCov().cast<T>();
    Eigen::Matrix<T, 2, 2> cnq = q->GetNormalCov().cast<T>();

    Eigen::Matrix<T, 2, 1> m1 = up - uq;
    Eigen::Matrix<T, 2, 1> m2 = unp + unq;
    Eigen::Matrix<T, 2, 2> c1 = cp + cq;
    Eigen::Matrix<T, 2, 2> c2 = cnp + cnq;

    T num = m1.dot(m2);
    T den = sqrt(m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace());
    *e = num / den;
    return true;
  }

  template <typename T>
  static Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw) {
    T cos_yaw = ceres::cos(yaw);
    T sin_yaw = ceres::sin(yaw);
    Eigen::Matrix<T, 2, 2> ret;
    ret << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
    return ret;
  }

  static ceres::CostFunction *Create(const SNDTCell *cellp,
                                     const SNDTCell *cellq) {
    return new ceres::AutoDiffCostFunction<SNDTCostFunctor, 1, 1, 1, 1>(
        new SNDTCostFunctor(cellp, cellq));
  }

  static double Score(const SNDTCell *p, const SNDTCell *q) {
    Eigen::Vector2d up = p->GetPointMean(), uq = q->GetPointMean();
    Eigen::Vector2d unp = p->GetNormalMean(), unq = q->GetNormalMean();
    Eigen::Matrix2d cp = p->GetPointCov(), cq = q->GetPointCov();
    Eigen::Matrix2d cnp = p->GetNormalCov(), cnq = q->GetNormalCov();
    Eigen::Vector2d m1 = up - uq, m2 = unp + unq;
    Eigen::Matrix2d c1 = cp + cq, c2 = cnp + cnq;
    double num = m1.dot(m2);
    double den = sqrt(m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace());
    return num / den;
  }

  static std::vector<double> MeanAndCov(const SNDTCell *p, const SNDTCell *q) {
    Eigen::Vector2d up = p->GetPointMean(), uq = q->GetPointMean();
    Eigen::Vector2d unp = p->GetNormalMean(), unq = q->GetNormalMean();
    Eigen::Matrix2d cp = p->GetPointCov(), cq = q->GetPointCov();
    Eigen::Matrix2d cnp = p->GetNormalCov(), cnq = q->GetNormalCov();
    Eigen::Vector2d m1 = up - uq, m2 = unp + unq;
    Eigen::Matrix2d c1 = cp + cq, c2 = cnp + cnq;
    return {m1.dot(m2), m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace()};
  }
};

Eigen::Affine2d SNDTMatch(const SNDTMap &target_map, const SNDTMap &source_map,
                          const SNDTParameters &params,
                          const Eigen::Affine2d &guess_tf) {
  auto kd = MakeKDTree(target_map.GetPoints());
  bool converge = false;
  int iteration = 0;
  auto cur_tf = guess_tf;
  int min_ctr = 0;
  double min_cost = std::numeric_limits<double>::max();

  while (!converge) {
    double x = 0, y = 0, t = 0;
    auto next_map = source_map.PseudoTransformCells(cur_tf);
    ceres::Problem problem;

    for (size_t i = 0; i < next_map.size(); ++i) {
      auto cellp = next_map[i];
      if (!cellp->HasGaussian()) continue;
      pcl::PointXY pt;
      pt.x = cellp->GetPointMean()(0), pt.y = cellp->GetPointMean()(1);
      std::vector<int> idx{0};
      std::vector<float> dist2{0};
      int found = kd.nearestKSearch(pt, 1, idx, dist2);
      if (!found) continue;
      auto cellq = target_map.GetCellForPoint(Eigen::Vector2d(
          kd.getInputCloud()->at(idx[0]).x, kd.getInputCloud()->at(idx[0]).y));
      if (!cellq || !cellq->HasGaussian()) continue;

      if (params.huber != 0) {
        ceres::LossFunction *loss(new ceres::HuberLoss(params.huber));
        problem.AddResidualBlock(SNDTCostFunctor::Create(cellp.get(), cellq),
                                 loss, &x, &y, &t);
      } else {
        problem.AddResidualBlock(SNDTCostFunctor::Create(cellp.get(), cellq),
                                 nullptr, &x, &y, &t);
      }
    }

    problem.SetParameterization(&t, AngleLocalParameterization::Create());

    ceres::Solver::Options options;
    if (params.verbose) options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = params.max_iterations;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cur_tf = Eigen::Rotation2Dd(t) * Eigen::Translation2d(x, y) * cur_tf;

    if (params.verbose) {
      std::cout << "Iteration " << iteration << ": " << std::endl;
      std::cout << summary.BriefReport() << std::endl;
    }

    // XXX: converge strategy
    if (min_cost < summary.final_cost) {
      ++min_ctr;
    } else {
      min_ctr = 0;
      min_cost = summary.final_cost;
    }
    converge = (min_ctr == 5) ||
               (Eigen::Vector2d(x, y).norm() < params.threshold) ||
               (iteration > params.max_iterations);

    // converge = true;
    ++iteration;
  }
  return cur_tf;
}
