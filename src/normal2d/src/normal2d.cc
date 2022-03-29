#include <bits/stdc++.h>
#include <normal2d/normal2d.h>
#include <pcl/search/kdtree.h>

// #define USE_OMP

#ifdef USE_OMP
#define THREADS 8
#include <omp.h>
#endif

inline pcl::KdTreeFLANN<pcl::PointXY> MakeKDTrees(
    const std::vector<Eigen::Vector2d> &points) {
  pcl::PointCloud<pcl::PointXY>::Ptr pc(new pcl::PointCloud<pcl::PointXY>);
  for (const auto &pt : points) {
    pcl::PointXY p;
    p.x = pt(0), p.y = pt(1);
    pc->push_back(p);
  }
  pcl::KdTreeFLANN<pcl::PointXY> ret;
  ret.setInputCloud(pc);
  return ret;
}

inline Eigen::Vector2d ComputeMeanWithIndices(
    const std::vector<Eigen::Vector2d> &matrices,
    const std::vector<int> &indices) {
  Eigen::Vector2d ret = Eigen::Vector2d::Zero();
  for (size_t i = 0; i < indices.size(); ++i) ret += matrices[indices[i]];
  ret /= indices.size();
  return ret;
}

inline Eigen::Matrix2d ComputeCovWithIndices(
    const std::vector<Eigen::Vector2d> &points,
    const std::vector<int> &indices,
    const Eigen::Vector2d &mean) {
  int n = indices.size();
  if (n <= 2) return Eigen::Matrix2d::Zero();
  Eigen::MatrixXd mp(2, n);
  for (int i = 0; i < n; ++i) mp.col(i) = points[indices[i]] - mean;
  Eigen::Matrix2d ret;
  ret = mp * mp.transpose() / (n - 1);
  return ret;
}

std::vector<Eigen::Vector2d> ComputeNormals(
    const std::vector<Eigen::Vector2d> &pc, double radius) {
  std::vector<Eigen::Vector2d> ret(pc.size());
  auto kd = MakeKDTrees(pc);

#ifdef USE_OMP
#pragma omp parallel for num_threads(THREADS)
#endif
  for (size_t i = 0; i < pc.size(); ++i) {
    if (!pc[i].allFinite()) {
      ret[i].fill(std::numeric_limits<double>::quiet_NaN());
      continue;
    }
    std::vector<int> indices;
    std::vector<float> dists;
    pcl::PointXY pt;
    pt.x = pc[i](0), pt.y = pc[i](1);
    int found = kd.radiusSearch(pt, radius, indices, dists, 0);
    if (!found || found == 1) {
      ret[i].fill(std::numeric_limits<double>::quiet_NaN());
      continue;
    }
    if (found == 2) {
      ret[i] = (pc[indices[0]] - pc[indices[1]]).unitOrthogonal();
    } else {
      Eigen::Vector2d mean = ComputeMeanWithIndices(pc, indices);
      Eigen::Matrix2d cov = ComputeCovWithIndices(pc, indices, mean);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> evd;
      ret[i] = evd.computeDirect(cov).eigenvectors().col(0);
    }
  }
  return ret;
}
