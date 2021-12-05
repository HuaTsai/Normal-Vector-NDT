#include <bits/stdc++.h>
#include <metric/metric.h>
#include <common/common.h>

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

int main(int argc, char **argv) {
  ros::init(argc, argv, "metric_test");
  ros::NodeHandle nh;
  auto pub1 = nh.advertise<nav_msgs::Path>("path1", 0, true);
  auto pub2 = nh.advertise<nav_msgs::Path>("path2", 0, true);
  auto pubg = nh.advertise<nav_msgs::Path>("pathg", 0, true);

  TrajectoryEvaluation te;
  te.set_evaltype(TrajectoryEvaluation::EvalType::kAbsolute);
  te.set_estpath(GetPath("src/metric/data/estndt.txt"));
  te.set_gtpath(GetPath("src/metric/data/gt.txt"));
  auto ate = te.ComputeRMSError2D();
  pub1.publish(te.estpath());
  pub2.publish(te.align_estpath());
  pubg.publish(te.gtpath());
  cout << "tlATE: " << ate.first.mean << ", " << ate.first.rms << endl;
  cout << "rotATE: " << ate.second.mean << ", " << ate.second.rms << endl;

  te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeBySingle);
  auto rte = te.ComputeRMSError2D();
  cout << "tlRTE: " << rte.first.mean << ", " << rte.first.rms << endl;
  cout << "rotRTE: " << rte.second.mean << ", " << rte.second.rms << endl;

  te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeByLength);
  te.set_length(30);
  auto rte2 = te.ComputeRMSError2D();
  cout << "tlRTE: " << rte2.first.mean << ", " << rte2.first.rms << endl;
  cout << "rotRTE: " << rte2.second.mean << ", " << rte2.second.rms << endl;

  ros::spin();
}
