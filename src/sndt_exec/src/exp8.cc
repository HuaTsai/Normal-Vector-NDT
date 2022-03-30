// Rabbit
#include <common/angle_utils.h>
#include <common/eigen_utils.h>
#include <common/other_utils.h>
#include <gtest/gtest.h>
#include <ndt/matcher.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

using namespace std;
using namespace Eigen;
using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;

int main() {
  // PointCloudType::Ptr source_pcl = PointCloudType::Ptr(new PointCloudType);
  // PointCloudType::Ptr target_pcl = PointCloudType::Ptr(new PointCloudType);
  // pcl::io::loadPCDFile<pcl::PointXYZ>(
  //     JoinPath(WSPATH, "src/ndt/data/bunny1.pcd"), *source_pcl);
  // pcl::io::loadPCDFile<pcl::PointXYZ>(
  //     JoinPath(WSPATH, "src/ndt/data/bunny2.pcd"), *target_pcl);
  // vector<Vector3d> source;
  // vector<Vector3d> target;
  // for (const auto &pt : *source_pcl)
  //   source.push_back(Vector3d(pt.x, pt.y, pt.z));
  // for (const auto &pt : *target_pcl)
  //   target.push_back(Vector3d(pt.x, pt.y, pt.z));

  // NDTMatcher m1(NDTMatcher::MatchType::kNDTLS, 1);
  // m1.SetSource(source);
  // m1.SetTarget(target);
  // auto r1 = m1.Align(guess);
  // auto e1 = TransNormRotDegAbsFromAffine3d(r1);
  // cout << e1(0) << " / " << e1(1) << " / " << m1.iteration() << endl;
  // m1.timer().Show();

  // NDTMatcher m2(NDTMatcher::MatchType::kNNDTLS, 1);
  // m2.SetSource(source);
  // m2.SetTarget(target);
  // auto r2 = m2.Align(guess);
  // auto e2 = TransNormRotDegAbsFromAffine3d(r2);
  // cout << e1(0) << " / " << e1(1) << " / " << m2.iteration() << endl;
  // m2.timer().Show();
}