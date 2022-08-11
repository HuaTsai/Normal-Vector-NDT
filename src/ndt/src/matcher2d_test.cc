#include <common/angle_utils.h>
#include <common/eigen_utils.h>
#include <gtest/gtest.h>
#include <ndt/matcher2d.h>

using namespace std;
using namespace Eigen;

class SinWaveTest : public ::testing::Test {
 protected:
  virtual void SetUp() override {
    for (double x = -3.8; x < 3.8; x += 0.05) {
      double y = 0.3 * (x + 4) * sin(x + 4) + 0.1;
      target.push_back(Vector2d(x, y));
    }
  }

  pair<Vector2d, double> Match(NDTMatcher2D &m, const Affine2d &offset) {
    auto source = TransformPoints(target, offset);
    m.SetSource(source);
    m.SetTarget(target);
    auto res = m.Align();
    auto diff = TransNormRotDegAbsFromAffine2d(res * offset);
    printf("etl: %.4f, erot: %.4f, iter: %d, opt: %.2f, ttl: %.2f\n", diff(0),
           diff(1), m.iteration(), m.timer().optimize() / 1000.,
           m.timer().total() / 1000.);
    auto tl = res.translation();
    auto ang = Rad2Deg(Rotation2Dd(res.rotation()).angle());
    return {tl, ang};
  }

  vector<Vector2d> target;
};

TEST_F(SinWaveTest, NDT) {
  auto m = NDTMatcher2D::GetBasic({kNDT, k1to1, kNoReject}, 0.5);
  Affine2d offset = Translation2d(-1, -1) * Rotation2Dd(Deg2Rad(5));
  auto [tl, ang] = Match(m, offset);
  EXPECT_NEAR(tl(0), 1, 0.1);
  EXPECT_NEAR(tl(1), 1, 0.1);
  EXPECT_NEAR(ang, -5, 0.1);
}

TEST_F(SinWaveTest, NNDT) {
  auto m = NDTMatcher2D::GetBasic({kNVNDT, k1to1, kNoReject}, 0.5, 0.5);
  Affine2d offset = Translation2d(-1, -1) * Rotation2Dd(Deg2Rad(5));
  auto [tl, ang] = Match(m, offset);
  EXPECT_NEAR(tl(0), 1, 0.1);
  EXPECT_NEAR(tl(1), 1, 0.1);
  EXPECT_NEAR(ang, -5, 0.1);
}
