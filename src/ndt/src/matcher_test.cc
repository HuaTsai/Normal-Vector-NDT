#include <common/angle_utils.h>
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
using namespace std::chrono_literals;

class BunnyTest : public ::testing::Test {
 protected:
  using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;
  virtual void SetUp() {
    source_pcl = PointCloudType::Ptr(new PointCloudType);
    target_pcl = PointCloudType::Ptr(new PointCloudType);
    pcl::io::loadPCDFile<pcl::PointXYZ>(
        JoinPath(WSPATH, "src/ndt/data/bunny1.pcd"), *source_pcl);
    pcl::io::loadPCDFile<pcl::PointXYZ>(
        JoinPath(WSPATH, "src/ndt/data/bunny2.pcd"), *target_pcl);
    for (const auto &pt : *source_pcl)
      source.push_back(Vector3d(pt.x, pt.y, pt.z));
    for (const auto &pt : *target_pcl)
      target.push_back(Vector3d(pt.x, pt.y, pt.z));
  }
  PointCloudType::Ptr source_pcl;
  PointCloudType::Ptr target_pcl;
  vector<Vector3d> source;
  vector<Vector3d> target;
};

// PCL ICP result is bad
TEST_F(BunnyTest, PCLICP) { EXPECT_TRUE(true); }

TEST_F(BunnyTest, PCLNDT) {
  PointCloudType::Ptr source_pcl2(new PointCloudType);
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> avg;
  avg.setLeafSize(0.2, 0.2, 0.2);
  avg.setInputCloud(source_pcl);
  avg.filter(*source_pcl2);
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  auto t1 = GetTime();
  ndt.setTransformationEpsilon(0.01);
  ndt.setStepSize(0.1);
  ndt.setResolution(1.0);
  ndt.setMaximumIterations(35);
  ndt.setInputSource(source_pcl2);
  ndt.setInputTarget(target_pcl);
  Matrix4f guess = (Translation3f(1.79387, 0.720047, 0) *
                    AngleAxisf(0.6931, Vector3f::UnitZ()))
                       .matrix();
  PointCloudType out;
  ndt.align(out, guess);
  auto t2 = GetTime();
  auto res = ndt.getFinalTransformation();
  Vector3f tl = res.block<3, 1>(0, 3);
  double ang = Rad2Deg(AngleAxisf(res.block<3, 3>(0, 0)).angle());
  printf("etl: %.4f, erot: %.4f, iter: %d, ttl: %.2f\n",
         (tl - Eigen::Vector3f(1, 1, 0)).norm(), abs(ang - 10.),
         ndt.getFinalNumIteration(), GetDiffTime(t1, t2) / 1000.);
  EXPECT_NEAR(tl(0), 1, 0.05);
  EXPECT_NEAR(tl(1), 1, 0.05);
  EXPECT_NEAR(tl(2), 0, 0.05);
  EXPECT_NEAR(ang, 10, 0.5);
}

TEST_F(BunnyTest, MyNDTLS) {
  NDTMatcher m(NDTMatcher::MatchType::kNDTLS, 1);
  m.SetSource(source);
  m.SetTarget(target);
  Affine3d guess = Translation3d(1.79387, 0.720047, 0) *
                   AngleAxisd(0.6931, Vector3d::UnitZ());
  auto res = m.Align(guess);
  Vector3d tl = res.translation();
  double ang = Rad2Deg(AngleAxisd(res.rotation()).angle());
  printf("etl: %.4f, erot: %.4f, iter: %d, opt: %.2f, ttl: %.2f\n",
         (tl - Eigen::Vector3d(1, 1, 0)).norm(), abs(ang - 10.), m.iteration(),
         m.timer().optimize() / 1000., m.timer().total() / 1000.);
  EXPECT_NEAR(tl(0), 1, 0.05);
  EXPECT_NEAR(tl(1), 1, 0.05);
  EXPECT_NEAR(tl(2), 0, 0.05);
  EXPECT_NEAR(ang, 10, 0.5);
}

TEST_F(BunnyTest, MyNNDTLS) {
  NDTMatcher m(NDTMatcher::MatchType::kNNDTLS, 1);
  m.SetSource(source);
  m.SetTarget(target);
  Affine3d guess = Translation3d(1.79387, 0.720047, 0) *
                   AngleAxisd(0.6931, Vector3d::UnitZ());
  auto res = m.Align(guess);
  Vector3d tl = res.translation();
  double ang = Rad2Deg(AngleAxisd(res.rotation()).angle());
  printf("etl: %.4f, erot: %.4f, iter: %d, opt: %.2f, ttl: %.2f\n",
         (tl - Eigen::Vector3d(1, 1, 0)).norm(), abs(ang - 10.), m.iteration(),
         m.timer().optimize() / 1000., m.timer().total() / 1000.);
  EXPECT_NEAR(tl(0), 1, 0.05);
  EXPECT_NEAR(tl(1), 1, 0.05);
  EXPECT_NEAR(tl(2), 0, 0.05);
  EXPECT_NEAR(ang, 10, 0.5);
}

TEST_F(BunnyTest, MyNDTTR) {
  NDTMatcher m(NDTMatcher::MatchType::kNDTTR, 1);
  m.SetSource(source);
  m.SetTarget(target);
  Affine3d guess = Translation3d(1.79387, 0.720047, 0) *
                   AngleAxisd(0.6931, Vector3d::UnitZ());
  auto res = m.Align(guess);
  Vector3d tl = res.translation();
  double ang = Rad2Deg(AngleAxisd(res.rotation()).angle());
  printf("etl: %.4f, erot: %.4f, iter: %d, opt: %.2f, ttl: %.2f\n",
         (tl - Eigen::Vector3d(1, 1, 0)).norm(), abs(ang - 10.), m.iteration(),
         m.timer().optimize() / 1000., m.timer().total() / 1000.);
  EXPECT_NEAR(tl(0), 1, 0.05);
  EXPECT_NEAR(tl(1), 1, 0.05);
  EXPECT_NEAR(tl(2), 0, 0.05);
  EXPECT_NEAR(ang, 10, 0.5);
}

TEST_F(BunnyTest, MyNNDTTR) {
  NDTMatcher m(NDTMatcher::MatchType::kNNDTTR, 1);
  m.SetSource(source);
  m.SetTarget(target);
  Affine3d guess = Translation3d(1.79387, 0.720047, 0) *
                   AngleAxisd(0.6931, Vector3d::UnitZ());
  auto res = m.Align(guess);
  Vector3d tl = res.translation();
  double ang = Rad2Deg(AngleAxisd(res.rotation()).angle());
  printf("etl: %.4f, erot: %.4f, iter: %d, opt: %.2f, ttl: %.2f\n",
         (tl - Eigen::Vector3d(1, 1, 0)).norm(), abs(ang - 10.), m.iteration(),
         m.timer().optimize() / 1000., m.timer().total() / 1000.);
  EXPECT_NEAR(tl(0), 1, 0.05);
  EXPECT_NEAR(tl(1), 1, 0.05);
  EXPECT_NEAR(tl(2), 0, 0.05);
  EXPECT_NEAR(ang, 10, 0.5);
}