#include <common/eigen_utils.h>
#include <gtest/gtest.h>

using namespace std;
using vec = Eigen::Vector2d;

TEST(Affine3dFromXYZRPY, BiDirection) {
  Eigen::Matrix<double, 6, 1> tf;
  tf << 1, 2, 3, 0.1, 0.5, 0.3;
  auto aff = Affine3dFromXYZRPY(tf);
  auto res = XYZRPYFromAffine3d(aff);
  for (int i = 0; i < 6; ++i) EXPECT_NEAR(res(i), tf(i), 1e-12);
}

TEST(ExcludeInfiniteInPlace, Basic) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  vector<vec> a{vec(0, 0), vec(nan, nan), vec(2, 2)};
  vector<vec> b{vec(nan, nan), vec(nan, nan), vec(2, 2)};
  ExcludeInfiniteInPlace(a, b);
  EXPECT_EQ((int)a.size(), 1);
  EXPECT_EQ((int)b.size(), 1);
  EXPECT_EQ(a[0], vec(2, 2));

  vector<vec> c{vec(0, 0), vec(1, 1), vec(2, 2), vec(3, 3)};
  vector<vec> d{vec(0, 0), vec(1, nan), vec(2, 2), vec(nan, 3)};
  ExcludeInfiniteInPlace(c, d);
  EXPECT_EQ((int)c.size(), 2);
  EXPECT_EQ((int)d.size(), 2);
  EXPECT_EQ(c[0], vec(0, 0));
  EXPECT_EQ(d[1], vec(2, 2));
}
