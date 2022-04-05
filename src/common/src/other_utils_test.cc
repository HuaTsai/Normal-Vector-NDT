#include <common/other_utils.h>
#include <gtest/gtest.h>

using namespace std;

TEST(LargestNIndices, Basic) {
  vector<double> data{10, 3, 2, 1, 7, 8, 32, 4, 7, 10};
  auto res = LargestNIndices(data, 5);
  vector<int> expect{6, 0, 9, 5, 4};
  EXPECT_EQ(res, expect);
}
