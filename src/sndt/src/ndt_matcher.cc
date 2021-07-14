#include "sndt/ndt_matcher.h"

#include <bits/stdc++.h>
#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_ros/point_cloud.h>

using namespace std;
using namespace Eigen;

struct CostFunction {
  NDTCell *p, *q;
  CostFunction(NDTCell *p_, NDTCell *q_) : p(p_), q(q_) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    T th = xyt[2];
    T m1[2] = {cos(th) * p->GetPointMean()(0) - sin(th) * p->GetPointMean()(1) +
                   xyt[0] - q->GetPointMean()(0),
               sin(th) * p->GetPointMean()(0) + cos(th) * p->GetPointMean()(1) +
                   xyt[1] - q->GetPointMean()(1)};
    T m2[2] = {cos(th) * p->GetNormalMean()(0) -
                   sin(th) * p->GetNormalMean()(1) + q->GetNormalMean()(0),
               sin(th) * p->GetNormalMean()(0) +
                   cos(th) * p->GetNormalMean()(1) + q->GetNormalMean()(1)};
    double ap = p->GetPointCov()(0, 0);
    double bp = p->GetPointCov()(0, 1);
    double cp = p->GetPointCov()(1, 1);
    double anp = p->GetNormalCov()(0, 0);
    double bnp = p->GetNormalCov()(0, 1);
    double cnp = p->GetNormalCov()(1, 1);
    double aq = q->GetPointCov()(0, 0);
    double bq = q->GetPointCov()(0, 1);
    double cq = q->GetPointCov()(1, 1);
    double anq = q->GetNormalCov()(0, 0);
    double bnq = q->GetNormalCov()(0, 1);
    double cnq = q->GetNormalCov()(1, 1);
    T c1[3] = {ap * cos(th) * cos(th) - 2. * bp * sin(th) * cos(th) +
                   cp * sin(th) * sin(th) + aq,
               ap * sin(th) * cos(th) - bp * sin(th) * sin(th) +
                   bp * cos(th) * cos(th) - cp * sin(th) * cos(th) + bq,
               ap * sin(th) * sin(th) + 2. * bp * sin(th) * cos(th) +
                   cp * cos(th) * cos(th) + cq};
    T c2[3] = {anp * cos(th) * cos(th) - 2. * bnp * sin(th) * cos(th) +
                   cnp * sin(th) * sin(th) + anq,
               anp * sin(th) * cos(th) - bnp * sin(th) * sin(th) +
                   bnp * cos(th) * cos(th) - cnp * sin(th) * cos(th) + bnq,
               anp * sin(th) * sin(th) + 2. * bnp * sin(th) * cos(th) +
                   cnp * cos(th) * cos(th) + cnq};
    T f = (m1[0] * m2[0] + m1[1] * m2[1]) * (m1[0] * m2[0] + m1[1] * m2[1]);
    T g = (c2[0] * m1[0] * m1[0] + 2. * c2[1] * m1[0] * m1[1] +
           c2[2] * m1[1] * m1[1]) +
          (c1[0] * m2[0] * m2[0] + 2. * c1[1] * m2[0] * m2[1] +
           c1[2] * m2[1] * m2[1]) +
          (c1[0] * c2[0] + 2. * c1[1] * c2[1] + c1[2] * c2[2]);
    T l = f / g;
    e[0] = l;
    return true;
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
    double num = m1.dot(m2) * m1.dot(m2);
    double den = m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace();
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
    return {m1.dot(m2) * m1.dot(m2),
            m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace()};
  }
};

// FIXME: check correctness
struct SymmCostFunction {
  NDTCell *p, *q;
  SymmCostFunction(NDTCell *p_, NDTCell *q_) : p(p_), q(q_) {}
  template <typename T>
  bool operator()(const T *const xyt, T *e) const {
    T th = xyt[2];
    T m1[2] = {cos(th) * p->GetPointMean()(0) - sin(th) * p->GetPointMean()(1) -
                   cos(th) * q->GetPointMean()(0) -
                   sin(th) * q->GetPointMean()(1) + xyt[0],
               sin(th) * p->GetPointMean()(0) + cos(th) * p->GetPointMean()(1) +
                   sin(th) * q->GetPointMean()(0) -
                   cos(th) * q->GetPointMean()(1) + xyt[1]};
    T m2[2] = {T(p->GetNormalMean()(0) + q->GetNormalMean()(0)),
               T(p->GetNormalMean()(1) + q->GetNormalMean()(1))};
    double ap = p->GetPointCov()(0, 0);
    double bp = p->GetPointCov()(0, 1);
    double cp = p->GetPointCov()(1, 1);
    double anp = p->GetNormalCov()(0, 0);
    double bnp = p->GetNormalCov()(0, 1);
    double cnp = p->GetNormalCov()(1, 1);
    double aq = q->GetPointCov()(0, 0);
    double bq = q->GetPointCov()(0, 1);
    double cq = q->GetPointCov()(1, 1);
    double anq = q->GetNormalCov()(0, 0);
    double bnq = q->GetNormalCov()(0, 1);
    double cnq = q->GetNormalCov()(1, 1);
    T c1[3] = {ap * cos(th) * cos(th) + aq * cos(th) * cos(th) -
                   2 * bp * sin(th) * cos(th) + 2 * bq * sin(th) * cos(th) +
                   cp * sin(th) * sin(th) + cq * sin(th) * sin(th),
               ap * sin(th) * cos(th) - aq * sin(th) * cos(th) -
                   bp * sin(th) * sin(th) + bp * cos(th) * cos(th) -
                   bq * sin(th) * sin(th) + bq * cos(th) * cos(th) -
                   cp * sin(th) * cos(th) + cq * sin(th) * cos(th),
               ap * sin(th) * sin(th) + aq * sin(th) * sin(th) +
                   2 * bp * sin(th) * cos(th) - 2 * bq * sin(th) * cos(th) +
                   cp * cos(th) * cos(th) + cq * cos(th) * cos(th)};
    T c2[3] = {T(anp + anq), T(bnp + bnq), T(cnp + cnq)};
    T f = (m1[0] * m2[0] + m1[1] * m2[1]) * (m1[0] * m2[0] + m1[1] * m2[1]);
    T g = (c2[0] * m1[0] * m1[0] + 2. * c2[1] * m1[0] * m1[1] +
           c2[2] * m1[1] * m1[1]) +
          (c1[0] * m2[0] * m2[0] + 2. * c1[1] * m2[0] * m2[1] +
           c1[2] * m2[1] * m2[1]) +
          (c1[0] * c2[0] + 2. * c1[1] * c2[1] + c1[2] * c2[2]);
    T l = f / g;
    // e[0] = -exp(l / 2.);
    e[0] = l;
    return true;
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
}

Matrix3d NDTMatcher::CeresMatch(NDTMap &target_map, NDTMap &source_map, const Matrix3d &guess_tf) {
  vmas.clear();
  auto kd = MakeKDTree(target_map);
  bool converge = false;
  double max_dist2 = numeric_limits<double>::max();
  double min_cost = numeric_limits<double>::max();
  int consec_min_ctr = 0;
  iteration_ = 0;
  auto cur_tf = guess_tf;

  while (!converge) {
    vector<pair<double, pair<NDTCell *, NDTCell *>>> dist2s;
    double xyt[3] = {0, 0, 0};
    auto nextNDT = source_map.PseudoTransformCells(cur_tf);
    ceres::Problem problem;

    int corres_cnt = 0;
    vector<MarkerArray> vma;
    for (auto cellp : nextNDT) {
      if (!cellp->BothHasGaussian()) continue;

      if (strategy_ == kDEFAULT) {
        auto cellq = target_map.GetClosestCellForPoint(cellp->GetPointMean(),
                                                        maxdist_of_cells_);
        if (!cellq || !cellq->BothHasGaussian()) continue;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
                new CostFunction(cellp.get(), cellq)),
            nullptr, xyt);
      } else if (strategy_ == kNEAREST_1_POINTS) {
        pcl::PointXYZ pt;
        pt.x = cellp->GetPointMean()(0);
        pt.y = cellp->GetPointMean()(1);
        pt.z = 0;
        vector<int> idx{0};
        vector<float> dist2{0};
        int found = kd.nearestKSearch(pt, 1, idx, dist2);
        if (!found || dist2.front() > max_dist2) continue;

        auto cellq = target_map.GetCellForPoint(
            Vector2d(kd.getInputCloud()->at(idx.at(0)).x,
                      kd.getInputCloud()->at(idx.at(0)).y));
        if (!cellq || !cellq->BothHasGaussian()) continue;
        dist2s.push_back({dist2.front(), {cellp.get(), cellq}});
      } else if (strategy_ == kUSE_CELLS_GREATER_THAN_TWO_POINTS) {
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
        vma.push_back(MarkerArrayOfCorrespondences(cellp.get(), cellq));
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
                new CostFunction(cellp.get(), cellq)),
            nullptr, xyt);
        ++corres_cnt;
      }
    }
    vmas.push_back(JoinMarkerArraysAndMarkers(vma));
    cout << "Iteration " << iteration_ << ": " << corres_cnt << "Corres" << endl;

    if (strategy_ == kNEAREST_1_POINTS) {
      sort(dist2s.begin(), dist2s.end(),
            [](auto a, auto b) { return a.first < b.first; });
      max_dist2 = min(max_dist2, dist2s[(dist2s.size() / 2)].first * 144);
      for (auto dist2cells : dist2s) {
        if (dist2cells.first > max_dist2) {
          break;
        }
        auto cellp = dist2cells.second.first;
        auto cellq = dist2cells.second.second;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CostFunction, 1, 3>(
                new CostFunction(cellp, cellq)),
            nullptr, xyt);
      }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 1;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (min_cost < summary.final_cost) {
      ++consec_min_ctr;
    } else {
      consec_min_ctr = 0;
      min_cost = summary.final_cost;
    }

    converge = (consec_min_ctr == 10) ||
                (Vector2d::Map(xyt, 2).norm() < threshold_) ||
                (iteration_ > max_iterations_);

    cur_tf = Rotation2Dd(xyt[2]) * Translation2d(xyt[0], xyt[1]) * cur_tf;
    ++iteration_;
  }
  return cur_tf;
}
