#include "sndt/ndt_matcher.h"

#include <bits/stdc++.h>
#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_ros/point_cloud.h>

using namespace std;
using namespace Eigen;

class AngleLocalParameterization {
 public:
  template <typename T>
  bool operator()(const T *t, const T *dt, T *t_add_dt) const {
    T pi(M_PI);
    T twopi(2. * M_PI);
    T sum(*t + *dt);
    *t_add_dt = sum - twopi * ceres::floor((sum + pi) / twopi);
    return true;
  }

  static ceres::LocalParameterization *Create() {
    return new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>;
  }
};

struct CostFunctor {
  NDTCell *p, *q;
  CostFunctor(NDTCell *p_, NDTCell *q_) : p(p_), q(q_) {}
  template <typename T>
  bool operator()(const T *const x, const T *const y, const T *const yaw, T *e) const {
    Matrix<T, 2, 2> R = RotationMatrix2D(*yaw);
    Matrix<T, 2, 1> t(*x, *y);

    Matrix<T, 2, 1> up = R * p->GetPointMean().cast<T>() + t;
    Matrix<T, 2, 1> unp = R * p->GetNormalMean().cast<T>();
    Matrix<T, 2, 2> cp = R * p->GetPointCov().cast<T>() * R.transpose();
    Matrix<T, 2, 2> cnp = R * p->GetNormalCov().cast<T>() * R.transpose();

    Matrix<T, 2, 1> uq = q->GetPointMean().cast<T>();
    Matrix<T, 2, 1> unq = q->GetNormalMean().cast<T>();
    Matrix<T, 2, 2> cq = q->GetPointCov().cast<T>();
    Matrix<T, 2, 2> cnq = q->GetNormalCov().cast<T>();

    Matrix<T, 2, 1> m1 = up - uq;
    Matrix<T, 2, 1> m2 = unp + unq;
    Matrix<T, 2, 2> c1 = cp + cq;
    Matrix<T, 2, 2> c2 = cnp + cnq;

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

  static ceres::CostFunction *Create(NDTCell *cellp, NDTCell *cellq) {
    return new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 1, 1>(new CostFunctor(cellp, cellq));
  }

  static double Score(NDTCell *p, NDTCell *q) {
    Vector2d up = p->GetPointMean();
    Vector2d uq = q->GetPointMean();
    Vector2d unp = p->GetNormalMean();
    Vector2d unq = q->GetNormalMean();
    Matrix2d cp = p->GetPointCov();
    Matrix2d cq = q->GetPointCov();
    Matrix2d cnp = p->GetNormalCov();
    Matrix2d cnq = q->GetNormalCov();
    Vector2d m1 = up - uq;
    Vector2d m2 = unp + unq;
    Matrix2d c1 = cp + cq;
    Matrix2d c2 = cnp + cnq;
    double num = m1.dot(m2);
    double den = sqrt(m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace());
    return num / den;
  }

  static vector<double> MeanAndCov(NDTCell *p, NDTCell *q) {
    Vector2d up = p->GetPointMean();
    Vector2d uq = q->GetPointMean();
    Vector2d unp = p->GetNormalMean();
    Vector2d unq = q->GetNormalMean();
    Matrix2d cp = p->GetPointCov();
    Matrix2d cq = q->GetPointCov();
    Matrix2d cnp = p->GetNormalCov();
    Matrix2d cnq = q->GetNormalCov();
    Vector2d m1 = up - uq;
    Vector2d m2 = unp + unq;
    Matrix2d c1 = cp + cq;
    Matrix2d c2 = cnp + cnq;
    return {m1.dot(m2), m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace()};
  }
};

pcl::KdTreeFLANN<pcl::PointXYZ> MakeKDTree(const NDTMap &map) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto cell : map)
    if (cell->BothHasGaussian())
      pc->push_back(
          pcl::PointXYZ(cell->GetPointMean()(0), cell->GetPointMean()(1), 0));
  pcl::KdTreeFLANN<pcl::PointXYZ> ret;
  ret.setInputCloud(pc);
  return ret;
}

NDTMatcher::NDTMatcher() {
  iteration_ = 0;
  max_iterations_ = 100;
  threshold_ = 0.01;
  maxdist_of_cells_ = 2;
  strategy_ = kNEAREST_1_POINTS;
  inlier_ratio_ = 1 / 2.;
  huber = 0;
  verbose = false;
}

Matrix3d NDTMatcher::CeresMatch(NDTMap &target_map, NDTMap &source_map, const Matrix3d &guess_tf) {
  vmas.clear();
  auto kd = MakeKDTree(target_map);
  bool converge = false;
  // double min_cost = numeric_limits<double>::max();
  // int consec_min_ctr = 0;
  iteration_ = 0;
  auto cur_tf = guess_tf;

  // vector<pair<NDTCell *, NDTCell *>> corres;
  vector<pair<int, NDTCell *>> corres;
  while (!converge) {
    double x = 0, y = 0, t = 0;
    auto nextNDT = source_map.PseudoTransformCells(cur_tf);
    ceres::Problem problem;

    for (size_t i = 0; i < nextNDT.size(); ++i) {
      auto cellp = nextNDT[i];
      if (!cellp->BothHasGaussian()) continue;

      if (strategy_ == kUSE_CELLS_GREATER_THAN_TWO_POINTS) {
        pcl::PointXYZ pt;
        pt.x = cellp->GetPointMean()(0), pt.y = cellp->GetPointMean()(1), pt.z = 0;
        vector<int> idx{0};
        vector<float> dist2{0};
        int found = kd.nearestKSearch(pt, 1, idx, dist2);
        if (!found) continue;
        auto cellq = target_map.GetCellForPoint(
            Vector2d(kd.getInputCloud()->at(idx.at(0)).x,
                      kd.getInputCloud()->at(idx.at(0)).y));
        if (!cellq || !cellq->BothHasGaussian()) continue;

        if (huber != 0) {
          ceres::LossFunction *loss(new ceres::HuberLoss(huber));
          problem.AddResidualBlock(CostFunctor::Create(cellp.get(), cellq), loss, &x, &y, &t);
        } else {
          problem.AddResidualBlock(CostFunctor::Create(cellp.get(), cellq), nullptr, &x, &y, &t);
        }
        corres.push_back({i, cellq});
      }
    }

    if (strategy_ == kUSE_CELLS_GREATER_THAN_TWO_POINTS) {
      problem.SetParameterization(&t, AngleLocalParameterization::Create());
    }

    // FIXME: heuristic?
    // problem.SetParameterLowerBound(x, 0, -0.3);
    // problem.SetParameterUpperBound(x, 0, 0.3);

    // problem.SetParameterLowerBound(y, 1, -0.1);
    // problem.SetParameterUpperBound(y, 1, 0.1);

    // problem.SetParameterLowerBound(t, 2, -0.05);
    // problem.SetParameterUpperBound(t, 2, 0.05);

    // FIXME: Marker For Debug
    vector<ceres::ResidualBlockId> vbki_before;
    problem.GetResidualBlocks(&vbki_before);
    vector<MarkerArray> vma_before;
    vector<double> cost_before;
    for (size_t i = 0; i < vbki_before.size(); ++i) {
      double cost;
      problem.EvaluateResidualBlock(vbki_before[i], true, &cost, nullptr, nullptr);
      cost_before.push_back(cost);
      char buf[10];
      sprintf(buf, "%.2f", cost);
      vma_before.push_back(MarkerArrayOfCorrespondences(nextNDT[corres[i].first].get(), corres[i].second, string(buf)));
    }

    ceres::Solver::Options options;
    if (verbose)
      options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cur_tf = Rotation2Dd(t) * Translation2d(x, y) * cur_tf;

    // FIXME: Marker For Debug
    auto nextNDT2 = source_map.PseudoTransformCells(cur_tf);
    double dcost = summary.initial_cost - summary.final_cost;
    vector<ceres::ResidualBlockId> vbki_after;
    problem.GetResidualBlocks(&vbki_after);
    vector<MarkerArray> vma_after;
    for (size_t i = 0; i < vbki_after.size(); ++i) {
      double cost;
      problem.EvaluateResidualBlock(vbki_after[i], true, &cost, nullptr, nullptr);
      char buf[30];
      double per = (cost_before[i] - cost) / dcost;
      sprintf(buf, "%.2f(%.1f%%)", cost, per);
      if (per > 0)
        vma_after.push_back(MarkerArrayOfCorrespondences(nextNDT2[corres[i].first].get(), corres[i].second, string(buf), Color::kLime));
      else
        vma_after.push_back(MarkerArrayOfCorrespondences(nextNDT2[corres[i].first].get(), corres[i].second, string(buf), Color::kRed));
    }
    vmas.push_back({JoinMarkerArrays(vma_before), JoinMarkerArrays(vma_after)});

    if (verbose) {
      cout << "Iteration " << iteration_ << ": " << endl;
      cout << summary.BriefReport() << endl;
    }
  
    // FIXME: converge strategy
    // if (min_cost < summary.final_cost) {
    //   ++consec_min_ctr;
    // } else {
    //   consec_min_ctr = 0;
    //   min_cost = summary.final_cost;
    // }

    // converge = (consec_min_ctr == 10) ||
    //             (Vector2d(x, y).norm() < threshold_) ||
    //             (iteration_ > max_iterations_);

    // FIXME: remove!
    converge = true;
    ++iteration_;
  }
  return cur_tf;
}
