#include <gtest/gtest.h>
#include <nndt/nmap.h>
#include <common/angle_utils.h>
using namespace std;
using namespace Eigen;

vector<Vector3d> EllipseData() {
  double x = 10, y = 6, xc = 0, yc = 0;
  vector<Vector3d> ret;
  for (int i = 0; i < 360; ++i) {
    double th = Deg2Rad(i);
    ret.push_back(Vector3d(xc + x * cos(th), yc + y * sin(th), 0));
  }
  return ret;
}

TEST(NMapTest, Ellipse) {
  auto data = EllipseData();
  NMap mp(1);
  mp.LoadPoints(data);
  EXPECT_EQ((int)mp.size(), 62);
  EXPECT_EQ(mp.at(Vector3i(1, 2, 0)).GetN(), 7);
  Vector3i idx;
  mp.SearchNearestCell(Eigen::Vector3d(-8, -3, 0), idx);
  EXPECT_EQ(idx, Vector3i(1, 2, 0));
}
