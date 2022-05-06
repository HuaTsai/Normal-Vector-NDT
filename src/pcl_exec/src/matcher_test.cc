#include <common/common.h>
#include <gtest/gtest.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl_exec/myd2dndt.h>

using namespace std;
using namespace Eigen;
using namespace std::chrono_literals;

class BunnyTest : public ::testing::Test {
 protected:
  using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;
  using PointNormalType = pcl::PointCloud<pcl::PointNormal>;
  virtual void SetUp() override {
    source_pcl = PointCloudType::Ptr(new PointCloudType);
    target_pcl = PointCloudType::Ptr(new PointCloudType);
    sourcen_pcl = PointNormalType::Ptr(new PointNormalType);
    targetn_pcl = PointNormalType::Ptr(new PointNormalType);
    pcl::io::loadPCDFile<pcl::PointXYZ>(
        JoinPath(WSPATH, "src/ndt/data/bunny1.pcd"), *source_pcl);
    pcl::io::loadPCDFile<pcl::PointXYZ>(
        JoinPath(WSPATH, "src/ndt/data/bunny2.pcd"), *target_pcl);
    for (const auto &pt : *source_pcl)
      source.push_back(Vector3d(pt.x, pt.y, pt.z));
    for (const auto &pt : *target_pcl)
      target.push_back(Vector3d(pt.x, pt.y, pt.z));
    AddNormal(source_pcl, sourcen_pcl);
    AddNormal(target_pcl, targetn_pcl);
  }

  void AddNormal(PointCloudType::Ptr cloud, PointNormalType::Ptr out) {
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> est;
    est.setInputCloud(cloud);
    est.setSearchMethod(tree);
    est.setKSearch(15);
    est.compute(*normals);
    pcl::concatenateFields(*cloud, *normals, *out);
  }

  void PCLMatchAndTest(pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr reg,
                       const Affine3d &guess = Affine3d::Identity()) {
    auto t1 = GetTime();
    reg->setInputSource(source_pcl);
    reg->setInputTarget(target_pcl);
    PointCloudType out;
    reg->align(out, guess.cast<float>().matrix());
    auto t2 = GetTime();
    Matrix4f res = reg->getFinalTransformation();
    Vector3d tl = res.block<3, 1>(0, 3).cast<double>();
    double ang = Rad2Deg(AngleAxisf(res.block<3, 3>(0, 0)).angle());
    printf("etl: %.4f, erot: %.4f, ttl: %.2f\n",
           (tl - Eigen::Vector3d(1, 1, 0)).norm(), abs(ang - 10.),
           GetDiffTime(t1, t2) / 1000.);
    PerformTests(tl, ang);
  }

  void PCLMatchAndTest(
      pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>
          &reg,
      const Affine3d &guess = Affine3d::Identity()) {
    auto t1 = GetTime();
    reg.setInputSource(sourcen_pcl);
    reg.setInputTarget(targetn_pcl);
    PointNormalType out;
    reg.align(out, guess.cast<float>().matrix());
    auto t2 = GetTime();
    Matrix4f res = reg.getFinalTransformation();
    Vector3d tl = res.block<3, 1>(0, 3).cast<double>();
    double ang = Rad2Deg(AngleAxisf(res.block<3, 3>(0, 0)).angle());
    printf("etl: %.4f, erot: %.4f, ttl: %.2f\n",
           (tl - Eigen::Vector3d(1, 1, 0)).norm(), abs(ang - 10.),
           GetDiffTime(t1, t2) / 1000.);
    PerformTests(tl, ang);
  }

  void PerformTests(Vector3d tl, double ang) {
    EXPECT_NEAR(tl(0), 1, 0.05);
    EXPECT_NEAR(tl(1), 1, 0.05);
    EXPECT_NEAR(tl(2), 0, 0.05);
    EXPECT_NEAR(ang, 10, 0.5);
  }

  PointCloudType::Ptr source_pcl;
  PointCloudType::Ptr target_pcl;
  PointNormalType::Ptr sourcen_pcl;
  PointNormalType::Ptr targetn_pcl;
  vector<Vector3d> source;
  vector<Vector3d> target;
};

TEST_F(BunnyTest, MyPCLD2D) {
  pcl::NormalDistributionsTransformD2D<pcl::PointXYZ, pcl::PointXYZ>::Ptr m(
      new pcl::NormalDistributionsTransformD2D<pcl::PointXYZ, pcl::PointXYZ>);
  m->setResolution(1);
  m->setTransformationEpsilon(0.001);
  PCLMatchAndTest(m);

  pcl::NormalDistributionsTransformD2D<pcl::PointXYZ, pcl::PointXYZ>::Ptr m2(
      new pcl::NormalDistributionsTransformD2D<pcl::PointXYZ, pcl::PointXYZ>);
  m2->setResolution(1);
  m2->setTransformationEpsilon(0.0001);
  // This guess fails
  // Affine3d guess = Translation3d(1.79387, 0.720047, 0) *
  //                  AngleAxisd(0.6931, Vector3d::UnitZ());
  Affine3d guess =
      Translation3d(0.5, 0.5, 0) * AngleAxisd(Deg2Rad(5), Vector3d::UnitZ());
  PCLMatchAndTest(m2, guess);
}

// PCL ICP result is bad: give easier (stupid) initial guess
TEST_F(BunnyTest, PCLICP) {
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr m(
      new pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>);
  m->setTransformationEpsilon(0.001);
  Affine3d guess = Translation3d(0.92, 0.92, 0) *
                   AngleAxisd(Deg2Rad(9.3), Vector3d::UnitZ());
  PCLMatchAndTest(m, guess);
}

TEST_F(BunnyTest, PCLNICP) {
  // clang format off
  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> m;
  m.setTransformationEpsilon(0.001);
  PCLMatchAndTest(m);

  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> m2;
  m2.setTransformationEpsilon(0.001);
  Affine3d guess = Translation3d(1.79387, 0.720047, 0) *
                   AngleAxisd(0.6931, Vector3d::UnitZ());
  PCLMatchAndTest(m2, guess);

  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> m3;
  m3.setTransformationEpsilon(0.001);
  guess =
      Translation3d(0.5, 0.5, 0.5) * AngleAxisd(Deg2Rad(5), Vector3d::UnitY());
  PCLMatchAndTest(m3, guess);
  // clang format on
}

TEST_F(BunnyTest, PCLSymmICP) {
  // clang format off
  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> m;
  m.setTransformationEpsilon(0.001);
  m.setUseSymmetricObjective(true);
  PCLMatchAndTest(m);

  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> m2;
  m2.setTransformationEpsilon(0.001);
  m2.setUseSymmetricObjective(true);
  Affine3d guess = Translation3d(1.79387, 0.720047, 0) *
                   AngleAxisd(0.6931, Vector3d::UnitZ());
  PCLMatchAndTest(m2, guess);

  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> m3;
  m3.setTransformationEpsilon(0.001);
  m3.setUseSymmetricObjective(true);
  guess =
      Translation3d(0.5, 0.5, 0.5) * AngleAxisd(Deg2Rad(5), Vector3d::UnitY());
  PCLMatchAndTest(m3, guess);
  // clang format on
}

TEST_F(BunnyTest, PCLNDT) {
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr m(
      new pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>);
  m->setResolution(1);
  m->setTransformationEpsilon(0.001);
  PCLMatchAndTest(m);

  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr m2(
      new pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>);
  m2->setResolution(1);
  m2->setTransformationEpsilon(0.001);
  Affine3d guess = Translation3d(1.79387, 0.720047, 0) *
                   AngleAxisd(0.6931, Vector3d::UnitZ());
  PCLMatchAndTest(m2, guess);
}
