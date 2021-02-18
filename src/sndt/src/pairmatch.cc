#include <bits/stdc++.h>
#include <rosbag/view.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_msgs/TFMessage.h>
#include <sensor_msgs/PointCloud2.h>
#include "sndt/ndt_matcher_2d.hpp"
#include "common/common.h"
#include "normal2d/Normal2dEstimation.h"
#include <pcl/features/normal_3d.h>
#include "sndt/ndt_conversions.hpp"

using namespace std;

vector<string> GetDataPath(string data) {
  vector<string> ret;
  if (data == "log24") {
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729278446231_scene-0299.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729298446271_scene-0300.bag");
    ret.push_back("/home/ee904/Desktop/Dataset/nuScenes/log24_1535729318549677_scene-0301.bag");
  }
  return ret;
}

vector<geometry_msgs::PoseStamped> GetPoses(const vector<string> &bag_paths) {
  vector<geometry_msgs::PoseStamped> ret;
  for (const auto &bag_path : bag_paths) {
    rosbag::Bag bag;
    bag.open(bag_path);
    for (rosbag::MessageInstance const m : rosbag::View(bag)) {
      tf2_msgs::TFMessage::ConstPtr tfmsg = m.instantiate<tf2_msgs::TFMessage>();
      if (tfmsg != nullptr) {
        auto tf = tfmsg->transforms.at(0);
        if (tf.header.frame_id == "map" && tf.child_frame_id == "car") {
          geometry_msgs::PoseStamped pst;
          pst.header = tf.header;
          pst.pose.position.x = tf.transform.translation.x;
          pst.pose.position.y = tf.transform.translation.y;
          pst.pose.position.z = tf.transform.translation.z;
          pst.pose.orientation = tf.transform.rotation;
          ret.push_back(pst);
        }
      }
    }
    bag.close();
  }
  return ret;
}

pcl::PointCloud<pcl::PointXY> ToPCLXY(const sensor_msgs::PointCloud2 &msg) {
  pcl::PointCloud<pcl::PointXYZI> pcxyzi;
  pcl::fromROSMsg(msg, pcxyzi);
  pcl::PointCloud<pcl::PointXY> ret;
  for (size_t i = 0; i < pcxyzi.points.size(); ++i) {
    pcl::PointXY pt;
    pt.x = pcxyzi.points.at(i).x;
    pt.y = pcxyzi.points.at(i).y;
    ret.points.push_back(pt);
  }
  return ret;
}

sensor_msgs::PointCloud2 AugmentPointCloud(
    const vector<sensor_msgs::PointCloud2> &pcs,
    const vector<geometry_msgs::PoseStamped> &poses,
    int start_frame_id, int frames) {
  sensor_msgs::PointCloud2 ret = pcs.at(start_frame_id);
  auto pose_t0 = common::GetPose(poses, ret.header.stamp);
  Eigen::Affine3d T_t0, T_tn;
  tf2::fromMsg(pose_t0, T_t0);
  for (int i = 1; i < frames; ++i) {
    sensor_msgs::PointCloud2 pctn = pcs.at(start_frame_id + i);
    tf2::fromMsg(common::GetPose(poses, pctn.header.stamp), T_tn);
    auto tsp = tf2::eigenToTransform(common::Conserve2DFromAffine3d(T_t0.inverse() * T_tn));
    tf2::doTransform(pctn, pctn, tsp);
    pcl::concatenatePointCloud(ret, pctn, ret);
  }
  ret.header.frame_id = "map";
  return ret;
}

visualization_msgs::MarkerArray NormalsMarkerArray(
    const pcl::PointCloud<pcl::PointXYZ> &pc,
    const pcl::PointCloud<pcl::PointXYZ> &normals) {
  visualization_msgs::MarkerArray ret;
  int nan_ctr = 0;
  for (size_t i = 0; i < pc.points.size(); ++i) {
    if (isnan(normals.at(i).x) || isnan(normals.at(i).y)) {
      ++nan_ctr;
      continue;
    }
    Eigen::Vector3d start(pc.points.at(i).x, pc.points.at(i).y, 0);
    Eigen::Vector3d end = start + Eigen::Vector3d(normals.at(i).x, normals.at(i).y, 0);
    ret.markers.push_back(common::MakeArrowMarkerByEnds(i, "map", start, end, common::Color::kLime));
  }
  cerr << "point size: " << pc.points.size() << endl;
  cerr << "nan normal: " << nan_ctr << endl;
  return ret;
}

Eigen::Matrix3d GetTransform(
    const vector<geometry_msgs::PoseStamped> &poses,
    const ros::Time &source_time,
    const ros::Time &target_time) {
  auto source_tf = common::GetPose(poses, source_time);
  auto target_tf = common::GetPose(poses, target_time);
  Eigen::Affine3d source_tf_e, target_tf_e;
  tf2::fromMsg(source_tf, source_tf_e);
  tf2::fromMsg(target_tf, target_tf_e);
  Eigen::Matrix4d Tgt4 = common::Conserve2DFromAffine3d(target_tf_e.inverse() * source_tf_e).matrix();
  Eigen::Matrix3d Tgt3 = Eigen::Matrix3d::Identity();
  Tgt3.block<2, 2>(0, 0) = Tgt4.block<2, 2>(0, 0);
  Tgt3.block<2, 1>(0, 2) = Tgt4.block<2, 1>(0, 3);
  return Tgt3;
}

void ComputeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr pc, double radius,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr output) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  Normal2dEstimation norm_estim;
  norm_estim.setInputCloud(pc);
  norm_estim.setSearchMethod(tree);
  norm_estim.setRadiusSearch(radius);
  norm_estim.compute(output);
}

double diff(Eigen::Matrix3d T1, Eigen::Matrix3d T2) {
  Eigen::Vector3d Tdiff(T1(0, 2) - T2(0, 2), T1(1, 2) - T2(1, 2), acos(T1(0, 0)) - acos(T2(0, 0)));
  return Tdiff.norm();
}

sndt::NDTMapMsg GenerateTransformMsg(pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                                     const Eigen::Matrix3d &T, double radius,
                                     double gridsize) {
  Eigen::Matrix4f mtx = Eigen::Matrix4f::Identity();
  mtx.block<2, 2>(0, 0) = T.block<2, 2>(0, 0).cast<float>();
  mtx.block<2, 1>(0, 3) = T.block<2, 1>(0, 2).cast<float>();
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc2(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*pc, *pc2, mtx);

  pcl::PointCloud<pcl::PointXYZ>::Ptr normal_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  ComputeNormals(pc2, radius, normal_cloud);
  NDTMap *map = new NDTMap(new LazyGrid2D(gridsize));
  map->loadPointCloud(*pc2, *normal_cloud);
  map->computeNDTCells();

  sndt::NDTMapMsg ret;
  toMessage(map, ret, "map");
  return ret;
}

int main(int argc, char **argv) {
  auto bag_paths = GetDataPath("log24");
  auto poses = GetPoses(bag_paths);
  vector<sensor_msgs::PointCloud2> pcs;
  common::SerializationInput("/home/ee904/Desktop/HuaTsai/NormalNDT/Research/pcs.ser", pcs);
  cout << "pcs size: " << pcs.size() << endl;

  int frames = 5;
  int target_id = 246;
  double radius = 1.5;
  double gridsize = 1;
  int source_id = target_id + frames;
  if (argc == 3) {
    target_id = atoi(argv[1]);
    source_id = target_id + atoi(argv[2]);
  }

  auto source_time = pcs.at(source_id).header.stamp;
  auto target_time = pcs.at(target_id).header.stamp;
  auto Tgt = GetTransform(poses, source_time, target_time);
  cout << "source time: " << source_time << endl;
  cout << "target time: " << target_time << endl;
  
  auto augspc = AugmentPointCloud(pcs, poses, source_id, frames);
  augspc.header.frame_id = "map";
  pcl::PointCloud<pcl::PointXYZ>::Ptr spc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr snormal_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(augspc, *spc);
  ComputeNormals(spc, radius, snormal_cloud);
  NDTMap source_map(new LazyGrid2D(gridsize));
  source_map.loadPointCloud(*spc, *snormal_cloud);
  source_map.computeNDTCells();
  pcl::io::savePCDFile(APATH(20210128/cases/spc00.pcd), *spc);

  auto augtpc = AugmentPointCloud(pcs, poses, target_id, frames);
  augtpc.header.frame_id = "map";
  pcl::PointCloud<pcl::PointXYZ>::Ptr tpc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tnormal_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(augtpc, *tpc);
  ComputeNormals(tpc, radius, tnormal_cloud);
  NDTMap target_map(new LazyGrid2D(gridsize));
  target_map.loadPointCloud(*tpc, *tnormal_cloud);
  target_map.computeNDTCells();
  pcl::io::savePCDFile(APATH(20210128/cases/tpc00.pcd), *tpc);
  common::WriteToFile(APATH(20210128/cases/tgt.txt), {Tgt(0, 2), Tgt(1, 2), acos(Tgt(0, 0))});

  NDTMatcherD2D_2D matcher;
  Eigen::Matrix3d Tout0, Tout1, Ttemp;
  Ttemp = Eigen::Matrix3d::Identity();
  matcher.ceresmatch(target_map, source_map, Ttemp);
  Tout0 = Ttemp;
  Ttemp = Eigen::Matrix3d::Identity();
  matcher.ceresmatchnn(target_map, source_map, Ttemp);
  Tout1 = Ttemp;
  cout << "result T0: " << endl << Tout0 << endl;
  cout << "result T1: " << endl << Tout1 << endl;
  cout << "actual T: " << endl << Tgt << endl;
  printf("T0: %.2f\n", diff(Tgt, Tout0));
  printf("T1: %.2f\n", diff(Tgt, Tout1));

  // Ttemp = Eigen::Matrix3d::Identity();
  // DoICP(spc, tpc, Ttemp);
  // DoICP2(spc, tpc, Ttemp);
  // DoSICP(spc, tpc, Ttemp);

  ros::init(argc, argv, "pairmatch");
  ros::NodeHandle nh;
  sndt::NDTMapMsg msg, msg2;
  toMessage(&source_map, msg, "map");
  toMessage(&target_map, msg2, "map");
  auto msg3 = GenerateTransformMsg(spc, Tout0, radius, gridsize);
  auto msg4 = GenerateTransformMsg(spc, Tout1, radius, gridsize);
  ros::Publisher pub_map = nh.advertise<sndt::NDTMapMsg>("map", 0, true);
  ros::Publisher pub_map2 = nh.advertise<sndt::NDTMapMsg>("map2", 0, true);
  ros::Publisher pub_map3 = nh.advertise<sndt::NDTMapMsg>("map3", 0, true);
  ros::Publisher pub_map4 = nh.advertise<sndt::NDTMapMsg>("map4", 0, true);
  ros::Publisher pub_pc = nh.advertise<sensor_msgs::PointCloud2>("pc", 0, true);
  ros::Publisher pub_pc2 = nh.advertise<sensor_msgs::PointCloud2>("pc2", 0, true);
  ros::Publisher pub_normal = nh.advertise<visualization_msgs::MarkerArray>("normal", 0, true);
  ros::Publisher pub_normal2 = nh.advertise<visualization_msgs::MarkerArray>("normal2", 0, true);

  pub_map.publish(msg);
  pub_map2.publish(msg2);
  pub_map3.publish(msg3);
  pub_map4.publish(msg4);
  pub_pc.publish(augspc);
  pub_pc2.publish(augtpc);
  pub_normal.publish(NormalsMarkerArray(*spc, *snormal_cloud));
  pub_normal2.publish(NormalsMarkerArray(*tpc, *tnormal_cloud));

  ros::spin();
}