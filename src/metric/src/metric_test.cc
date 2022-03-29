#include <bits/stdc++.h>
#include <common/common.h>
#include <gtest/gtest.h>
#include <metric/metric.h>

using namespace std;

nav_msgs::Path GetPath(string file) {
  ifstream fin(JoinPath(WSPATH, file));
  nav_msgs::Path ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  double time, x, y, z, qx, qy, qz, qw;
  while (fin >> time >> x >> y >> z >> qx >> qy >> qz >> qw) {
    geometry_msgs::PoseStamped pst;
    pst.header.stamp = ros::Time(time);
    pst.pose.position.x = x;
    pst.pose.position.y = y;
    pst.pose.position.z = z;
    pst.pose.orientation.w = qw;
    pst.pose.orientation.x = qx;
    pst.pose.orientation.y = qy;
    pst.pose.orientation.z = qz;
    ret.poses.push_back(pst);
  }
  return ret;
}

TEST(TrajectoryEvaluation, TrajectoryEvaluation) {
  ros::Time::init();
  double toler = 0.01;
  TrajectoryEvaluation te;
  te.set_evaltype(TrajectoryEvaluation::EvalType::kAbsolute);
  te.set_estpath(GetPath("src/metric/data/estndt.txt"));
  te.set_gtpath(GetPath("src/metric/data/gt.txt"));
  auto ate = te.ComputeRMSError2D();
  EXPECT_NEAR(ate.first.rms, 9.5176, toler);
  EXPECT_NEAR(ate.second.rms, 5.9714, toler);
  te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeBySingle);
  auto rte = te.ComputeRMSError2D();
  EXPECT_NEAR(rte.first.rms, 0.0861, toler);
  EXPECT_NEAR(rte.second.rms, 0.3118, toler);
  te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeByLength);
  te.set_length(30);
  auto rte2 = te.ComputeRMSError2D();
  EXPECT_NEAR(rte2.first.rms, 0.7915, toler);
  EXPECT_NEAR(rte2.second.rms, 2.0352, toler);
}
