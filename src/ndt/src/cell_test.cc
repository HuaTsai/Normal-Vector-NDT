#include <gtest/gtest.h>
#include <ndt/cell.h>

using namespace Eigen;
using namespace std;
using v = Vector3d;

TEST(CellTest, Basic) {
  Cell cell;
  // Not Initialized
  EXPECT_EQ(cell.GetCellType(), Cell::CellType::kNoInit);

  // No Points
  cell.ComputeGaussian();
  EXPECT_EQ(cell.GetCellType(), Cell::CellType::kFewPoints);

  cell.AddPoint(v(0, 0, 0));
  cell.AddPoint(v(1, 1, 1));
  cell.AddPoint(v(1, 0, 1));
  cell.ComputeGaussian();
  EXPECT_EQ(cell.GetCellType(), Cell::CellType::kFewPoints);

  cell.AddPoint(v(0, 1, 2));
  cell.AddPoint(v(2, 1, 0));
  cell.ComputeGaussian();
  EXPECT_EQ(cell.GetCellType(), Cell::CellType::kFewPoints);

  cell.AddPoint(v(1, 1, 3));
  cell.ComputeGaussian();
  EXPECT_EQ(cell.GetCellType(), Cell::CellType::kRegular);
}

TEST(CellTest, Line) {
  // clang-format off
  Cell cell;
  cell.SetPoints({v(0, 0, 0), v(1, 1, 1), v(2, 2, 2), v(3, 3, 3), v(4, 4, 4), v(5, 5, 5)});
  cell.ComputeGaussian();
  EXPECT_EQ(cell.GetCellType(), Cell::CellType::kLine);

  cell.SetPoints({v(0, 0, 0), v(1, 1, 1), v(2, 2, 1.9999999), v(3, 3, 3), v(4, 4, 4), v(5, 5, 5)});
  cell.ComputeGaussian();
  EXPECT_EQ(cell.GetCellType(), Cell::CellType::kLine);
  // clang-format on
}

TEST(CellTest, Plane) {
  // clang-format off
  Cell cell;
  cell.SetPoints({v(0, 0, 0), v(1, 1, 0), v(3, 2, 0), v(10, 2, 0), v(9, 1, 0), v(9, 7, 0)});
  cell.ComputeGaussian();
  EXPECT_EQ(cell.GetCellType(), Cell::CellType::kPlane);

  cell.SetPoints({v(0, 0, 0), v(1, 1, 0), v(3, 2, 0), v(10, 2, 0), v(9, 1, 0), v(9, 7, 0.000001)});
  cell.ComputeGaussian();
  EXPECT_EQ(cell.GetCellType(), Cell::CellType::kPlane);
  // clang-format on
}
