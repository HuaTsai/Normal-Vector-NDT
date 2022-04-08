#include <common/angle_utils.h>
#include <gtest/gtest.h>
#include <ndt/nmap2d.h>

using namespace std;
using namespace Eigen;

vector<Vector2d> MakeFakeData() {
  vector<Vector2d> ret;
  for (double x = -3.8; x < 3.8; x += 0.05) {
    double y = 0.3 * (x + 4) * sin(x + 4) + 0.1;
    ret.push_back(Vector2d(x, y));
  }
  return ret;
}

TEST(NMap2D, Basic) {
  NMap2D mp(0.5);
  mp.LoadPoints(MakeFakeData());
  EXPECT_EQ((int)mp.size(), 27);
  EXPECT_EQ(mp.at(Vector2i(0, 3)).GetN(), 7);
  Vector2i idx;
  mp.SearchNearestCell(Vector2d(0, -0.5), idx);
  EXPECT_EQ(idx(0), 7);
  EXPECT_EQ(idx(1), 1);
  auto cells = mp.SearchCellsInRadius(Eigen::Vector2d(0, -0.5), 0.5);
  EXPECT_EQ((int)cells.size(), 3);
}
