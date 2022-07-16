#include <common/common.h>
#include <gtest/gtest.h>
#include <ndt/matcher.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

using namespace std;
using namespace Eigen;
using namespace std::chrono_literals;

class BunnyTest : public ::testing::Test {
 protected:
  using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;
  virtual void SetUp() override {
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
    sample_guess = Translation3d(1.79387, 0.720047, 0) *
                   AngleAxisd(0.6931, Vector3d::UnitZ());
  }

  void MatchAndTest(Matcher &m, const Affine3d &guess = Affine3d::Identity()) {
    m.SetSource(source);
    m.SetTarget(target);
    auto res = m.Align(guess);
    Vector3d tl = res.translation();
    double ang = Rad2Deg(AngleAxisd(res.rotation()).angle());
    printf("etl: %.4f, erot: %.4f, iter: %d, opt: %.2f, ttl: %.2f\n",
           (tl - Eigen::Vector3d(1, 1, 0)).norm(), abs(ang - 10.),
           m.iteration(), m.timer().optimize() / 1000.,
           m.timer().total() / 1000.);
    PerformTests(tl, ang);
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

  void PerformTests(Vector3d tl, double ang) {
    EXPECT_NEAR(tl(0), 1, 0.05);
    EXPECT_NEAR(tl(1), 1, 0.05);
    EXPECT_NEAR(tl(2), 0, 0.05);
    EXPECT_NEAR(ang, 10, 0.5);
  }

  PointCloudType::Ptr source_pcl;
  PointCloudType::Ptr target_pcl;
  vector<Vector3d> source;
  vector<Vector3d> target;
  Affine3d sample_guess;
};

// PCL ICP result is bad: give easier (stupid) initial guess
TEST_F(BunnyTest, PCLICP) {
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr m(
      new pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>);
  m->setTransformationEpsilon(0.001);
  Affine3d guess = Translation3d(0.92, 0.92, 0) *
                   AngleAxisd(Deg2Rad(9.3), Vector3d::UnitZ());
  PCLMatchAndTest(m, guess);
}

TEST_F(BunnyTest, MyICP) {
  auto m = ICPMatcher::GetBasic({});
  MatchAndTest(m);
  auto m2 = ICPMatcher::GetBasic({});
  MatchAndTest(m2, sample_guess);
}

TEST_F(BunnyTest, MySICP) {
  auto m = SICPMatcher::GetBasic({}, 1);
  MatchAndTest(m);
  auto m2 = SICPMatcher::GetBasic({}, 1);
  MatchAndTest(m2, sample_guess);
}

TEST_F(BunnyTest, MySICPOMP) {
  auto m = SICPMatcher::GetBasic({kUseNormalOMP}, 1);
  MatchAndTest(m);
  auto m2 = SICPMatcher::GetBasic({kUseNormalOMP}, 1);
  MatchAndTest(m2, sample_guess);
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
  PCLMatchAndTest(m2, sample_guess);
}

TEST_F(BunnyTest, MyNDT) {
  auto m = NDTMatcher::GetBasic({kNDT, k1to1}, 1);
  MatchAndTest(m);
  auto m2 = NDTMatcher::GetBasic({kNDT, k1to1}, 1);
  MatchAndTest(m2, sample_guess);
}

TEST_F(BunnyTest, MyNNDT) {
  auto m = NDTMatcher::GetBasic({kNNDT, k1to1}, 1);
  MatchAndTest(m);
  auto m2 = NDTMatcher::GetBasic({kNNDT, k1to1}, 1);
  MatchAndTest(m2, sample_guess);
}

TEST_F(BunnyTest, MyNDTAnalytic) {
  auto m = NDTMatcher::GetBasic({kNDT, k1to1, kAnalytic}, 1);
  MatchAndTest(m);
  auto m2 = NDTMatcher::GetBasic({kNDT, k1to1, kAnalytic}, 1);
  MatchAndTest(m2, sample_guess);
}

TEST_F(BunnyTest, MyNNDTAnalytic) {
  auto m = NDTMatcher::GetBasic({kNNDT, k1to1, kAnalytic}, 1);
  MatchAndTest(m);
  auto m2 = NDTMatcher::GetBasic({kNNDT, k1to1, kAnalytic}, 1);
  MatchAndTest(m2, sample_guess);
}

TEST_F(BunnyTest, MyNDTIterative) {
  auto m = NDTMatcher::GetIter({kNDT, k1to1}, {0.5, 1, 2});
  MatchAndTest(m);
  auto m2 = NDTMatcher::GetIter({kNDT, k1to1}, {0.5, 1, 2});
  MatchAndTest(m2, sample_guess);
}

TEST_F(BunnyTest, MyNNDTIterative) {
  auto m = NDTMatcher::GetIter({kNNDT, k1to1}, {0.5, 1, 2});
  MatchAndTest(m);
  auto m2 = NDTMatcher::GetIter({kNNDT, k1to1}, {0.5, 1, 2});
  MatchAndTest(m2, sample_guess);
}

TEST_F(BunnyTest, MyNNDTIterativeAnalytic) {
  auto m = NDTMatcher::GetIter({kNNDT, k1to1, kAnalytic}, {0.5, 1, 2});
  MatchAndTest(m);
  auto m2 = NDTMatcher::GetIter({kNNDT, k1to1, kAnalytic}, {0.5, 1, 2});
  MatchAndTest(m2, sample_guess);
}

class BunnyTest2 : public ::testing::Test {
 protected:
  using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;
  virtual void SetUp() override {
    source_pcl = PointCloudType::Ptr(new PointCloudType);
    pcl::io::loadOBJFile<pcl::PointXYZ>(
        JoinPath(WSPATH, "src/ndt/data/bunny.obj"), *source_pcl);
    for (const auto &pt : *source_pcl)
      source.push_back(10 * Vector3d(pt.x, pt.y, pt.z));
    cout << source.size() << endl;
    auto aff =
        Affine3dFromXYZRPY({1, 1, 1, Deg2Rad(5), Deg2Rad(5), Deg2Rad(5)});
    target = TransformPoints(source, aff);
    gt_tl << 1, 1, 1;
    gt_ang = Rad2Deg(AngleAxisd(aff.rotation()).angle());
  }

  void MatchAndTest(Matcher &m, const Affine3d &guess = Affine3d::Identity()) {
    m.SetSource(source);
    m.SetTarget(target);
    auto res = m.Align(guess);
    Vector3d tl = res.translation();
    double ang = Rad2Deg(AngleAxisd(res.rotation()).angle());
    printf("etl: %.4f, erot: %.4f, iter: %d, opt: %.2f, ttl: %.2f\n",
           (tl - gt_tl).norm(), abs(ang - gt_ang), m.iteration(),
           m.timer().optimize() / 1000., m.timer().total() / 1000.);
    PerformTests(tl, ang);
  }

  void PerformTests(Vector3d tl, double ang) {
    EXPECT_NEAR(tl(0), gt_tl(0), 0.05);
    EXPECT_NEAR(tl(1), gt_tl(1), 0.05);
    EXPECT_NEAR(tl(2), gt_tl(2), 0.05);
    EXPECT_NEAR(ang, gt_ang, 0.5);
  }

  PointCloudType::Ptr source_pcl;
  vector<Vector3d> source;
  vector<Vector3d> target;
  Vector3d gt_tl;
  double gt_ang;
};

TEST_F(BunnyTest2, D2DNDT) {
  auto m = NDTMatcher::GetBasic({kNDT, k1to1, kAnalytic}, 1);
  MatchAndTest(m);
}

TEST_F(BunnyTest2, NVNDT) {
  auto m = NDTMatcher::GetBasic({kNNDT, k1to1, kAnalytic}, 1);
  MatchAndTest(m);
}
