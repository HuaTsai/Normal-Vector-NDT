/********************************************************
This is the implementation of radar ego-motion on Nuscenes dataset following
paper "Instantaneous Ego-Motion Estimation using Multiple Doppler Radars", in
2014 IEEE International Conference on Robotics and Automation (ICRA) , 2014, pp.
1592-1597.
-------------------------------
INPUT Topic: (conti_radar::Measurement)
  /radar_front
  /radar_front_right
  /radar_front_left
  /radar_back_right
  /radar_back_left
-------------------------------
OUTPUT Topic:
  Velocity: (nav_msgs::Odometry)
    /vel
  Inliers: (conti_radar::Measurement)
    /radar_front_inlier
    /radar_front_right_inlier
    /radar_front_left_inlier
    /radar_back_right_inlier
    /radar_back_left_inlier
  Outliers: (conti_radar::Measurement)
    /radar_front_outlier
    /radar_front_right_outlier
    /radar_front_left_outlier
    /radar_back_right_outlier
    /radar_back_left_outlier
-------------------------------
by Frank Kung 2019 Dec
-------------------------------
This implemntation is futher refactored and modified for storing results.
New INPUT Topic:
  Save std::vector<common::EgoPointClouds> to outpath: (std_msgs::Empty)
    /savepc (usage: rostopic pub /savepc std_msgs/Empty -1)
New OUTPUT Topic:
  Augment point cloud: (sensor_msgs::PointCloud2)
    /merge_pc
-------------------------------
by Adam Huang 2021 August
*********************************************************/
#include <bits/stdc++.h>
#include <conti_radar/Measurement.h>
#include <geometry_msgs/Twist.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_eigen/tf2_eigen.h>
#include <boost/program_options.hpp>
#include <common/EgoPointClouds.h>
#include <common/common.h>

using namespace std;
using namespace literals::chrono_literals;
using namespace Eigen;
namespace po = boost::program_options;

ros::Publisher pub, pub_pc;
ros::Publisher f_outlier_pub;
ros::Publisher fr_outlier_pub;
ros::Publisher fl_outlier_pub;
ros::Publisher br_outlier_pub;
ros::Publisher bl_outlier_pub;

ros::Publisher f_inlier_pub;
ros::Publisher fr_inlier_pub;
ros::Publisher fl_inlier_pub;
ros::Publisher br_inlier_pub;
ros::Publisher bl_inlier_pub;

vector<common::EgoPointClouds> vepcs;
string outpath;

unordered_map<string, ros::Publisher> publishers;
unordered_map<string, Vector3d> sensor_origins;
unordered_map<string, Matrix4d> sensor_tfs;
unordered_map<string, Matrix<double, 2, 3>> S;

typedef message_filters::sync_policies::ApproximateTime<
    conti_radar::Measurement, conti_radar::Measurement,
    conti_radar::Measurement, conti_radar::Measurement,
    conti_radar::Measurement>
    NoCloudSyncPolicy;

sensor_msgs::PointCloud2 conti_to_rospc(const conti_radar::Measurement &msg) {
  sensor_msgs::PointCloud2 ret;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < msg.points.size(); ++i) {
    pcl::PointXYZ point;
    if (msg.points.at(i).invalid_state == 0x00) {
      point.x = msg.points.at(i).longitude_dist;
      point.y = msg.points.at(i).lateral_dist;
      point.z = 0;
      // point.intensity = msg.points.at(i).rcs;
      pc->points.push_back(point);
    }
  }
  pcl::toROSMsg(*pc, ret);
  ret.header = msg.header;
  ret.is_dense = true;
  return ret;
}

void SystemMotion(string id, const conti_radar::MeasurementConstPtr &input,
                  MatrixXd &Rj, VectorXd &VDj) {
  int size = input->points.size();
  double beta = sensor_origins[id](2);
  MatrixXd Mj(size, 2);
  VDj.resize(size, 1);
  for (int i = 0; i < size; ++i) {
    double x = input->points[i].longitude_dist;
    double y = input->points[i].lateral_dist;
    double vx = input->points[i].longitude_vel;
    double vy = input->points[i].lateral_vel;
    double theta = atan2(y, x);
    Mj(i, 0) = cos(theta + beta);
    Mj(i, 1) = sin(theta + beta);
    VDj(i, 0) = -Vector2d(vx, vy).dot(Vector2d(x, y).normalized());
  }
  Rj = Mj * S[id];
}

vector<int> InliersAfterRANSAC(const MatrixXd &R, const VectorXd &VD, double threshold) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 0; i < R.rows(); ++i) {
		pcl::PointXYZ pt;
		pt.x = R(i, 0);
		pt.y = R(i, 1);
		pt.z = VD(i, 0);
		cloud->push_back(pt);
  }

  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model(
      new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud));
  vector<int> inliers;
  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model);
  ransac.setMaxIterations(2000);
  ransac.setDistanceThreshold(threshold);
  ransac.computeModel();
  ransac.getInliers(inliers);
  return inliers;
}

MatrixXd PreserveRows(const MatrixXd &mtx, const vector<int> &cols) {
  MatrixXd ret(cols.size(), mtx.cols());
  for (int i = 0; i < ret.rows(); ++i)
    ret.row(i) = mtx.row(cols[i]);
  return ret;
}

Matrix3d ComputeCovariance(const MatrixXd &RANSAC_R, const MatrixXd &RANSAC_VD, const Vector3d &ego_mat) {
  VectorXd err = (RANSAC_R * ego_mat) - RANSAC_VD;
  MatrixXd scalar = ((err.transpose() * err) / (RANSAC_R.rows() - 3));
  return scalar(0, 0) * (RANSAC_R.transpose() * RANSAC_R).inverse();
}

nav_msgs::Odometry MakeOdom(double time, const Vector3d &ego_mat, const Matrix3d &cov_mat) {
  nav_msgs::Odometry odom;
  odom.header.frame_id = "/car";
  odom.header.stamp = ros::Time(time);
  odom.twist.twist.linear.x = ego_mat(1, 0);
  odom.twist.twist.linear.y = ego_mat(2, 0);
  odom.twist.twist.angular.z = ego_mat(0, 0);
  odom.twist.covariance[0] = cov_mat(0, 0);
  odom.twist.covariance[1] = cov_mat(0, 1);
  odom.twist.covariance[5] = cov_mat(0, 2);
  odom.twist.covariance[6] = cov_mat(1, 0);
  odom.twist.covariance[7] = cov_mat(1, 1);
  odom.twist.covariance[11] = cov_mat(1, 2);
  odom.twist.covariance[30] = cov_mat(2, 0);
  odom.twist.covariance[31] = cov_mat(2, 1);
  odom.twist.covariance[35] = cov_mat(2, 2);
  odom.twist.covariance[14] = 99999;
  odom.twist.covariance[21] = 99999;
  odom.twist.covariance[28] = 99999;
  return odom;
}

conti_radar::Measurement ContiMsg(
    const conti_radar::MeasurementConstPtr& input, const vector<int> &indices) {
  conti_radar::Measurement ret;
  ret.header = input->header;
  for (const auto &index : indices)
    ret.points.push_back(input->points.at(index));
  return ret;
}

conti_radar::Measurement ContiMsgComp(
    string id, const conti_radar::MeasurementConstPtr &input,
    const vector<int> &indices, const Vector3d &ego_mat) {
  conti_radar::Measurement ret;
  ret.header = input->header;
  Vector2d vj = Rotation2Dd(-sensor_origins[id](2)) * (S[id] * ego_mat);
  for (const auto &index : indices) {
    conti_radar::ContiRadar msg = input->points.at(index);
    msg.longitude_vel = msg.longitude_vel + vj(0);
    msg.lateral_vel = msg.lateral_vel + vj(1);
    ret.points.push_back(msg);
  }
  return ret;
}

Matrix4d GetTransform(const Vector3d &xyt) {
  return (Translation3d(xyt(0), xyt(1), 0) * AngleAxisd(xyt(2), Vector3d::UnitZ())).matrix();
}

vector<geometry_msgs::Point> ToGeometryMsgs(const sensor_msgs::PointCloud2 &msg) {
  vector<geometry_msgs::Point> ret;
  pcl::PointCloud<pcl::PointXYZ> pc;
  pcl::fromROSMsg(msg, pc);
  transform(pc.begin(), pc.end(), back_inserter(ret), [](const auto &p){
    geometry_msgs::Point pt;
    pt.x = p.x, pt.y = p.y, pt.z = p.z;
    return pt;
  });
  return ret;
}

common::PointCloudSensor MakePointCloudSensor(string frame_id,
                                              const Matrix4d &origin,
                                              const sensor_msgs::PointCloud2 &pc,
                                              sensor_msgs::PointCloud2 &augpc) {
  sensor_msgs::PointCloud2 temp;
  pcl_ros::transformPointCloud(origin.cast<float>(), pc, temp);
  pcl::concatenatePointCloud(augpc, temp, augpc);
  common::PointCloudSensor ret;
  ret.id = frame_id;
  ret.origin = tf2::toMsg(Affine3d(origin));
  ret.points = ToGeometryMsgs(pc);
  return ret;
}

void Callback(const conti_radar::MeasurementConstPtr& f_input,
              const conti_radar::MeasurementConstPtr& fl_input,
              const conti_radar::MeasurementConstPtr& fr_input,
              const conti_radar::MeasurementConstPtr& bl_input,
              const conti_radar::MeasurementConstPtr& br_input) {
  double f_t = f_input->header.stamp.toSec();
  double fl_t = fl_input->header.stamp.toSec();
  double fr_t = fr_input->header.stamp.toSec();
  double bl_t = bl_input->header.stamp.toSec();
  double br_t = br_input->header.stamp.toSec();
  vector<double> times = {f_t, fl_t, fr_t, bl_t, br_t};
  double min_t = *min_element(times.begin(), times.end());

  MatrixXd Rf, Rfr, Rfl, Rbr, Rbl;
  VectorXd VDf, VDfr, VDfl, VDbr, VDbl;
  SystemMotion("f", f_input, Rf, VDf);
  SystemMotion("fr", fr_input, Rfr, VDfr);
  SystemMotion("fl", fl_input, Rfl, VDfl);
  SystemMotion("br", br_input, Rbr, VDbr);
  SystemMotion("bl", bl_input, Rbl, VDbl);

  vector<int> sizes = {(int)Rf.rows(), (int)Rfr.rows(), (int)Rfl.rows(),
                       (int)Rbr.rows(), (int)Rbl.rows()};
  vector<int> partialsum_sizes;
  partial_sum(sizes.begin(), sizes.end(), back_inserter(partialsum_sizes));
  int total_size = accumulate(sizes.begin(), sizes.end(), 0);
  MatrixXd R(total_size, 3);
  VectorXd VD(total_size);
  R << Rf, Rfr, Rfl, Rbr, Rbl;
  VD << VDf, VDfr, VDfl, VDbr, VDbl;

  vector<int> inliers = InliersAfterRANSAC(R, VD, 0.27);
  MatrixXd RANSAC_R = PreserveRows(R, inliers);
  VectorXd RANSAC_VD = PreserveRows(VD, inliers);
  Vector3d ego_mat = RANSAC_R.bdcSvd(ComputeThinU | ComputeThinV).solve(RANSAC_VD);
  Matrix3d cov_mat = ComputeCovariance(RANSAC_R, RANSAC_VD, ego_mat);
  nav_msgs::Odometry odom = MakeOdom(min_t, ego_mat, cov_mat);

  vector<int> f_inliers, fr_inliers, fl_inliers, br_inliers, bl_inliers;
  vector<int> f_outliers, fr_outliers, fl_outliers, br_outliers, bl_outliers;
  int j = 0;
  for (int i = 0; i < total_size; ++i) {
    if (inliers[j] == i) {
      if (i < partialsum_sizes[0])
        f_inliers.push_back(i);
      else if (i < partialsum_sizes[1])
        fr_inliers.push_back(i - partialsum_sizes[0]);
      else if (i < partialsum_sizes[2])
        fl_inliers.push_back(i - partialsum_sizes[1]);
      else if (i < partialsum_sizes[3])
        br_inliers.push_back(i - partialsum_sizes[2]);
      else if (i < partialsum_sizes[4])
        bl_inliers.push_back(i - partialsum_sizes[3]);
      ++j;
    } else {
      if (i < partialsum_sizes[0])
        f_outliers.push_back(i);
      else if (i < partialsum_sizes[1])
        fr_outliers.push_back(i - partialsum_sizes[0]);
      else if (i < partialsum_sizes[2])
        fl_outliers.push_back(i - partialsum_sizes[1]);
      else if (i < partialsum_sizes[3])
        br_outliers.push_back(i - partialsum_sizes[2]);
      else if (i < partialsum_sizes[4])
        bl_outliers.push_back(i - partialsum_sizes[3]);
    }
  }

  conti_radar::Measurement f_in, fr_in, fl_in, br_in, bl_in;
  f_in = ContiMsg(f_input, f_inliers);
  fr_in = ContiMsg(fr_input, fr_inliers);
  fl_in = ContiMsg(fl_input, fl_inliers);
  br_in = ContiMsg(br_input, br_inliers);
  bl_in = ContiMsg(bl_input, bl_inliers);

  conti_radar::Measurement f_out, fr_out, fl_out, br_out, bl_out;
  f_out = ContiMsgComp("f", f_input, f_outliers, ego_mat);
  fr_out = ContiMsgComp("fr", fr_input, fr_outliers, ego_mat);
  fl_out = ContiMsgComp("fl", fl_input, fl_outliers, ego_mat);
  br_out = ContiMsgComp("br", br_input, br_outliers, ego_mat);
  bl_out = ContiMsgComp("bl", bl_input, bl_outliers, ego_mat);

  common::EgoPointClouds epcs;
  sensor_msgs::PointCloud2 augpc;

  epcs.stamp = ros::Time(min_t);
  epcs.vxyt = {ego_mat(1), ego_mat(2), ego_mat(0)};

  Vector3d vxyt = Vector3d::Map(epcs.vxyt.data(), 3);
  Matrix4d tf_td;
  
  tf_td = GetTransform(vxyt * (f_t - min_t)) * sensor_tfs["f"];
  auto pcsr_f = MakePointCloudSensor("front", tf_td, conti_to_rospc(f_in), augpc);
  epcs.pcs.push_back(pcsr_f);

  tf_td = GetTransform(vxyt * (fr_t - min_t)) * sensor_tfs["fr"];
  auto pcsr_fr = MakePointCloudSensor("front right", tf_td, conti_to_rospc(fr_in), augpc);
  epcs.pcs.push_back(pcsr_fr);

  tf_td = GetTransform(vxyt * (fl_t - min_t)) * sensor_tfs["fl"];
  auto pcsr_fl = MakePointCloudSensor("front left", tf_td, conti_to_rospc(fl_in), augpc);
  epcs.pcs.push_back(pcsr_fl);

  tf_td = GetTransform(vxyt * (br_t - min_t)) * sensor_tfs["br"];
  auto pcsr_br = MakePointCloudSensor("back right", tf_td, conti_to_rospc(br_in), augpc);
  epcs.pcs.push_back(pcsr_br);

  tf_td = GetTransform(vxyt * (bl_t - min_t)) * sensor_tfs["bl"];
  auto pcsr_bl = MakePointCloudSensor("back left", tf_td, conti_to_rospc(bl_in), augpc);
  epcs.pcs.push_back(pcsr_bl);

  epcs.augpc = ToGeometryMsgs(augpc);
  vepcs.push_back(epcs);

  augpc.header.stamp = ros::Time(min_t);
  augpc.header.frame_id = "/car";

  publishers["f_out"].publish(f_out);
  publishers["fr_out"].publish(fr_out);
  publishers["fl_out"].publish(fl_out);
  publishers["br_out"].publish(br_out);
  publishers["bl_out"].publish(bl_out);

  publishers["f_in"].publish(f_in);
  publishers["fr_in"].publish(fr_in);
  publishers["fl_in"].publish(fl_in);
  publishers["br_in"].publish(br_in);
  publishers["bl_in"].publish(bl_in);

  pub.publish(odom);
  pub_pc.publish(augpc);
}

void savecb(const std_msgs::Empty::ConstPtr &msg) {
  SerializationOutput(outpath, vepcs);
  ROS_INFO("Saving... Done");
  vepcs.clear();
}

void PreComputeSystem() {
  for (const auto &s : sensor_origins) {
    MatrixXd Sj(2, 3);
    Sj << -s.second(1), 1., 0., s.second(0), 0., 1.;
    S.insert({s.first, Sj});
    sensor_tfs.insert({s.first, GetTransform(s.second)});
  }
}

int main(int argc, char** argv) {
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("outpath", po::value<string>(&outpath)->required(), "EgoPointClouds folder path");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  sensor_origins["f"] = Vector3d(3.41199994087, 0, 0.003);
  sensor_origins["fr"] = Vector3d(2.42199993134, -0.800000011921, -1.588);
  sensor_origins["fl"] = Vector3d(2.42199993134, 0.800000011921, 1.542);
  sensor_origins["br"] = Vector3d(-0.561999976635, -0.617999970913, -3.074);
  sensor_origins["bl"] = Vector3d(-0.561999976635, 0.628000020981, 3.044);

  PreComputeSystem();

  ros::init(argc, argv, "radar_ego_motion");
  ros::NodeHandle nh;

  message_filters::Subscriber<conti_radar::Measurement> sub_f(nh, "/radar_front", 1);
  message_filters::Subscriber<conti_radar::Measurement> sub_fl(nh, "/radar_front_left", 1);
  message_filters::Subscriber<conti_radar::Measurement> sub_fr(nh, "/radar_front_right", 1);
  message_filters::Subscriber<conti_radar::Measurement> sub_bl(nh, "/radar_back_left", 1);
  message_filters::Subscriber<conti_radar::Measurement> sub_br(nh, "/radar_back_right", 1);

  message_filters::Synchronizer<NoCloudSyncPolicy>* no_cloud_sync_;

  no_cloud_sync_ = new message_filters::Synchronizer<NoCloudSyncPolicy>(
      NoCloudSyncPolicy(3), sub_f, sub_fl, sub_fr, sub_bl, sub_br);
  no_cloud_sync_->registerCallback(boost::bind(&Callback, _1, _2, _3, _4, _5));
  ros::Subscriber sub_save = nh.subscribe<std_msgs::Empty>("savepc", 0, savecb);

  pub = nh.advertise<nav_msgs::Odometry>("vel", 1);
  pub_pc = nh.advertise<sensor_msgs::PointCloud2>("merge_pc", 1);

  publishers["f_out"] = nh.advertise<conti_radar::Measurement>("radar_front_outlier", 1);
  publishers["fr_out"] = nh.advertise<conti_radar::Measurement>("radar_front_right_outlier", 1);
  publishers["fl_out"] = nh.advertise<conti_radar::Measurement>("radar_front_left_outlier", 1);
  publishers["br_out"] = nh.advertise<conti_radar::Measurement>("radar_back_right_outlier", 1);
  publishers["bl_out"] = nh.advertise<conti_radar::Measurement>("radar_back_left_outlier", 1);

  publishers["f_in"] = nh.advertise<conti_radar::Measurement>("radar_front_inlier", 1);
  publishers["fr_in"] = nh.advertise<conti_radar::Measurement>("radar_front_right_inlier", 1);
  publishers["fl_in"] = nh.advertise<conti_radar::Measurement>("radar_front_left_inlier", 1);
  publishers["br_in"] = nh.advertise<conti_radar::Measurement>("radar_back_right_inlier", 1);
  publishers["bl_in"] = nh.advertise<conti_radar::Measurement>("radar_back_left_inlier", 1);

  ros::spin();
}
