/********************************************************
This is the implementation of radar ego-motion on Nuscenes dataset following
paper "Instantaneous Ego-Motion Estimation using Multiple Doppler Radars", in
2014 IEEE International Conference on Robotics and Automation (ICRA) , 2014, pp.
1592-1597.
-------------------------------
INPUT Topic: (conti_radar::Measurement)
        /radar_front  /radar_front_right  /radar_front_left  /radar_back_right
/radar_back_left
-------------------------------
OUTPUT Topic:
  Velocity: (nav_msgs::Odometry)
                /vel
  Inliers: (conti_radar::Measurement)
                /radar_front_inlier  /radar_front_right_inlier
/radar_front_left_inlier  /radar_back_right_inlier  /radar_back_left_inlier
  Outliers: (conti_radar::Measurement)
                /radar_front_outlier  /radar_front_right_outlier
/radar_front_left_outlier  /radar_back_right_outlier  /radar_back_left_outlier
-------------------------------
by Frank Kung 2019 Dec
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
#include "sndt/conti_tool.hpp"
#include "sndt/EgoPointClouds.h"
#include "common/EgoPointClouds.h"
#include "common/common.h"
#include "dbg/dbg.h"

using namespace std;
using namespace std::literals::chrono_literals;
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

ros::Publisher f_vel_pub;
ros::Publisher fr_vel_pub;
ros::Publisher fl_vel_pub;
ros::Publisher br_vel_pub;
ros::Publisher bl_vel_pub;

ros::Publisher f_vel_c_pub;
ros::Publisher fr_vel_c_pub;
ros::Publisher fl_vel_c_pub;
ros::Publisher br_vel_c_pub;
ros::Publisher bl_vel_c_pub;

vector<common::EgoPointClouds> vepcs;
string outfolder;

//-- debug --//
int total_size = 0;

typedef message_filters::sync_policies::ApproximateTime<
    conti_radar::Measurement, conti_radar::Measurement,
    conti_radar::Measurement, conti_radar::Measurement,
    conti_radar::Measurement>
    NoCloudSyncPolicy;

bool comp(int a, int b) { return (a < b); }

vector<geometry_msgs::Point> PointCloudAdapter(
    const sensor_msgs::PointCloud2 &msg) {
  vector<geometry_msgs::Point> ret;
  pcl::PointCloud<pcl::PointXYZ> pc;
  pcl::fromROSMsg(msg, pc);
  for (auto &p : pc.points) {
    geometry_msgs::Point pt;
    pt.x = p.x; pt.y = p.y; pt.z = p.z;
    ret.push_back(pt);
  }
  return ret;
}

Eigen::Matrix4f getTransform(double x, double y, double theta) {
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform(0, 0) = cos(theta);
  transform(0, 1) = -sin(theta);
  transform(1, 0) = sin(theta);
  transform(1, 1) = cos(theta);
  transform(0, 3) = x;
  transform(1, 3) = y;
  return transform;
}

Eigen::Matrix4f getTdTransform(double td, double vx, double vy, double vth) {
  return getTransform(vx * td, vy * td, vth / 180. * M_PI * td);
}

std::vector<int> calcRansacMat(Eigen::MatrixXd VD_mat, Eigen::MatrixXd R_mat,
                               double threshold) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);

  cloud->width = R_mat.rows();
  cloud->height = 1;
  cloud->is_dense = true;

  for (int i = 0; i < R_mat.rows(); ++i) {
		pcl::PointXYZ pt;
		pt.x = R_mat(i, 0);
		pt.y = R_mat(i, 1);
		pt.z = VD_mat(i, 0);
		cloud->points.push_back(pt);
  }

  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
      new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud));

  std::vector<int> inliers;
  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
  ransac.setMaxIterations(2000);
  ransac.setDistanceThreshold(threshold);
  ransac.computeModel();
  ransac.getInliers(inliers);
  // pcl::copyPointCloud(*cloud, inliers, *final);

  // pcl::visualization::PCLVisualizer::Ptr viewer;
  // viewer = simpleVis(cloud);
  // while (!viewer->wasStopped()) {
  //   viewer->spinOnce(100);
  //   std::this_thread::sleep_for(100ms);
  // }

  return inliers;
}

Eigen::MatrixXd deleteCols(Eigen::MatrixXd mat, std::vector<int> cols) {
  Eigen::MatrixXd rm_mat(cols.size(), mat.cols());
  for (int i = 0; i < rm_mat.rows(); i++) rm_mat.row(i) = mat.row(cols[i]);
  return rm_mat;
}

void radar_sensor(const conti_radar::MeasurementConstPtr& input, float beta,
                  float r_x, float r_y, Eigen::MatrixXd& M_mat,
                  Eigen::MatrixXd& R_mat, Eigen::MatrixXd& VD_mat) {
  double valid_size = 0;
  // for(int i=0;i<(input->points.size());i++)if
  // ((input->points[i].invalid_state) == 0x00){valid_size++;}
  valid_size = input->points.size();

  cout << "size: " << valid_size << " ";

  total_size = total_size + valid_size;

  VD_mat.resize(valid_size, 1);  // N x 1
  M_mat.resize(valid_size, 2);   // N x 2
  R_mat.resize(valid_size, 3);   // N x 3

  int n = 0;
  for (size_t i = 0; i < (input->points.size()); i++) {
    // if ((input->points[i].invalid_state) == 0x00){
    double theta;
    double x = input->points[i].longitude_dist;
    double y = input->points[i].lateral_dist;
    double vx = input->points[i].longitude_vel;
    double vy = input->points[i].lateral_vel;
    theta = atan2(y, x);
    theta = theta + beta;
    M_mat(n, 0) = cos(theta);
    M_mat(n, 1) = sin(theta);

    Eigen::Vector2d radial_vector(-x, -y);
    Eigen::Vector2d radial_vector_normalized = radial_vector.normalized();
    Eigen::Vector2d vel(vx, vy);
    double vel_radial = vel.dot(radial_vector_normalized);  // projection

    VD_mat(n, 0) = vel_radial;
    n++;
    //}
  }

  Eigen::MatrixXd S_mat;
  //-- for general condition (omega,vx,vy)--//
  S_mat.resize(2, 3);  // 2 x 3
  S_mat(0, 0) = -r_y;
  S_mat(0, 1) = 1.0;
  S_mat(0, 2) = 0.0;
  S_mat(1, 0) = r_x;
  S_mat(1, 1) = 0.0;
  S_mat(1, 2) = 1.0;
  R_mat = M_mat * S_mat;  // N x 3
  /*
          //-- for ackerman condition (omega,vx)--//
          S_mat.resize(2, 2); // 2 x 2
          S_mat(0,0) = -r_y; S_mat(0,1) = 1.0;
          S_mat(1,0) = r_x;  S_mat(1,1) = 0.0;
          R_mat = M_mat*S_mat; // N x 2
  */
}

conti_radar::Measurement contiMsgOutlierFilter(
    const conti_radar::MeasurementConstPtr& input, vector<int> outliers) {
  conti_radar::Measurement conti_msg_array;
  conti_msg_array.header = input->header;
  for (size_t i = 0; i < outliers.size(); ++i) {
    conti_msg_array.points.push_back(input->points.at(outliers.at(i)));
  }
  return conti_msg_array;
}

conti_radar::Measurement contiMsgOutlierFilterComp(
    const conti_radar::MeasurementConstPtr& input, vector<int> outliers,
    float beta, Eigen::MatrixXd vj) {
  conti_radar::Measurement conti_msg_array;
  conti_msg_array.header = input->header;

  tf::Transform radar_to_center;
  tf::Quaternion q;
  q.setRPY(0, 0, -beta);
  radar_to_center.setRotation(q);

  tf::Transform radar_v_from_center;
  radar_v_from_center.setOrigin(tf::Vector3(vj(0, 0), vj(1, 0), 0));

  tf::Transform radar_v_from_radar = radar_to_center * radar_v_from_center;

  for (size_t i = 0; i < outliers.size(); ++i) {
    conti_radar::ContiRadar point_msg = input->points.at(i);
    point_msg.longitude_vel = point_msg.longitude_vel + radar_v_from_radar.getOrigin().x();
    point_msg.lateral_vel = point_msg.lateral_vel + radar_v_from_radar.getOrigin().y();
    conti_msg_array.points.push_back(point_msg);
  }
  return conti_msg_array;
}

conti_radar::Measurement contiMsgInlierFilter(
    const conti_radar::MeasurementConstPtr& input, vector<int> inliers) {
  conti_radar::Measurement conti_msg_array;
  conti_msg_array.header = input->header;
  for (size_t i = 0; i < inliers.size(); ++i) {
    conti_msg_array.points.push_back(input->points.at(inliers.at(i)));
  }
  return conti_msg_array;
}

vector<int> outlierToInlier(vector<int> outlier, int size) {
  vector<int> inlier;
  size_t j = 0;
  for (int i = 0; i < size; ++i) {
    if (j < outlier.size() && i == outlier.at(j)) {
      ++j;
    } else {
      inlier.push_back(i);
    }
  }
  return inlier;
}

Eigen::MatrixXd calcVj(Eigen::MatrixXd A, Eigen::MatrixXd y) {  // y = Ax
  Eigen::MatrixXd x;
  x.resize(2, 1);
  if (A.rows() > 2 && y.rows() > 2) {
    x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
    // cout << "solution: " << x(0,0) << ", " << x(1,0) << endl;
  }
  return x;
}

Eigen::MatrixXd calcVjxVjy(Eigen::MatrixXd ego_mat, float r_x, float r_y) {
  Eigen::MatrixXd Vj;
  Eigen::MatrixXd S_mat;
  S_mat.resize(2, 3);  // 2 x 3
  S_mat(0, 0) = -r_y;
  S_mat(0, 1) = 1.0;
  S_mat(0, 2) = 0.0;
  S_mat(1, 0) = r_x;
  S_mat(1, 1) = 0.0;
  S_mat(1, 2) = 1.0;
  Vj = S_mat * ego_mat;
  return Vj;
}

Eigen::MatrixXd getCovMat(Eigen::MatrixXd RANSAC_R, Eigen::MatrixXd RANSAC_VD,
                          Eigen::MatrixXd ego_mat) {
  Eigen::MatrixXd cov_mat;
  cov_mat.resize(3, 3);  // 3 x 3

  Eigen::MatrixXd err = (RANSAC_R * ego_mat) - RANSAC_VD;
  Eigen::MatrixXd cov_mat_scalar = ((err.transpose() * err) / (RANSAC_R.rows() - 3));
  cov_mat = cov_mat_scalar(0, 0) * (RANSAC_R.transpose() * RANSAC_R).inverse();

  // cout << "cov_mat: \n" << cov_mat << endl;
  /*
          cout << "sig^T*sig: \n" << err.transpose()*err << endl;
          cout << "(RANSAC_R.rows() - 3): \n" << (RANSAC_R.rows() - 3) << endl;
          cout << "cov_mat_scalar: \n" << cov_mat_scalar << endl;
          cout << "R^T*R: \n" << (RANSAC_R.transpose()*RANSAC_R).inverse() <<
     endl;
  */
  return cov_mat;
}

nav_msgs::Odometry MakeOdom(double min_t, const Eigen::MatrixXd& ego_mat,
                            const Eigen::MatrixXd& cov_mat) {
  nav_msgs::Odometry odom;
  odom.header.frame_id = "/car";
  odom.header.stamp = ros::Time(min_t);
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

common::PointCloudSensor MakePointCloudSensor(string frame_id,
                                              const Eigen::Matrix4f& origin,
                                              const sensor_msgs::PointCloud2& pc,
                                              sensor_msgs::PointCloud2& augpc) {
  sensor_msgs::PointCloud2 temp;
  pcl_ros::transformPointCloud(origin, pc, temp);
  pcl::concatenatePointCloud(augpc, temp, augpc);
  common::PointCloudSensor ret;
  ret.id = frame_id;
  ret.origin = tf2::toMsg(common::Affine3dFromMatrix4f(origin));
  ret.points = PointCloudAdapter(pc);
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
  double list[] = {f_t, fl_t, fr_t, bl_t, br_t};
  double min_t = *std::min_element(list, list + 5);

  float f_x = 3.41199994087;  // 3.6
  float f_y = 0.0;
  float beta_f = 0.003;
  Eigen::Matrix4f tf_f = getTransform(f_x, f_y, beta_f);
  Eigen::MatrixXd M_mat_f, R_mat_f, VD_mat_f;
  radar_sensor(f_input, beta_f, f_x, f_y, M_mat_f, R_mat_f, VD_mat_f);

  float fr_x = 2.42199993134;
  float fr_y = -0.800000011921;
  float beta_fr = -1.588;
  Eigen::Matrix4f tf_fr = getTransform(fr_x, fr_y, beta_fr);
  Eigen::MatrixXd M_mat_fr, R_mat_fr, VD_mat_fr;
  radar_sensor(fr_input, beta_fr, fr_x, fr_y, M_mat_fr, R_mat_fr, VD_mat_fr);

  float fl_x = 2.42199993134;
  float fl_y = 0.800000011921;
  float beta_fl = 1.542;
  Eigen::Matrix4f tf_fl = getTransform(fl_x, fl_y, beta_fl);
  Eigen::MatrixXd M_mat_fl, R_mat_fl, VD_mat_fl;
  radar_sensor(fl_input, beta_fl, fl_x, fl_y, M_mat_fl, R_mat_fl, VD_mat_fl);

  float br_x = -0.561999976635;
  float br_y = -0.617999970913;
  float beta_br = -3.074;
  Eigen::Matrix4f tf_br = getTransform(br_x, br_y, beta_br);
  Eigen::MatrixXd M_mat_br, R_mat_br, VD_mat_br;
  radar_sensor(br_input, beta_br, br_x, br_y, M_mat_br, R_mat_br, VD_mat_br);

  float bl_x = -0.561999976635;
  float bl_y = 0.628000020981;
  float beta_bl = 3.044;
  Eigen::Matrix4f tf_bl = getTransform(bl_x, bl_y, beta_bl);
  Eigen::MatrixXd M_mat_bl, R_mat_bl, VD_mat_bl;
  radar_sensor(bl_input, beta_bl, bl_x, bl_y, M_mat_bl, R_mat_bl, VD_mat_bl);

  int f_size = R_mat_f.rows();
  int fr_size = R_mat_fr.rows();
  int fl_size = R_mat_fl.rows();
  int br_size = R_mat_br.rows();
  int bl_size = R_mat_bl.rows();

  Eigen::MatrixXd R(f_size + fr_size + fl_size + br_size + bl_size, R_mat_f.cols());
  R << R_mat_f, R_mat_fr, R_mat_fl, R_mat_br, R_mat_bl;
  Eigen::MatrixXd VD(f_size + fr_size + fl_size + br_size + bl_size, VD_mat_f.cols());
  VD << VD_mat_f, VD_mat_fr, VD_mat_fl, VD_mat_br, VD_mat_bl;

  std::vector<int> inliers;
  inliers = calcRansacMat(VD, R, 0.27);  // 0.27
  Eigen::MatrixXd RANSAC_R = deleteCols(R, inliers);
  Eigen::MatrixXd RANSAC_VD = deleteCols(VD, inliers);
  Eigen::MatrixXd ego_mat =
      RANSAC_R.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
          .solve(RANSAC_VD);
  Eigen::MatrixXd cov_mat = getCovMat(RANSAC_R, RANSAC_VD, ego_mat);

  nav_msgs::Odometry odom = MakeOdom(min_t, ego_mat, cov_mat);

  //-- Outliers Msg (comp) --//
  std::vector<int> f_outliers, fr_outliers, fl_outliers, br_outliers, bl_outliers;

  int j = 0;
  for (int i = 0; i < R.rows(); i++) {
    if (inliers[j] == i) {
      j++;
    } else {
      if (i < f_size) {
        f_outliers.push_back(i);
      } else if (f_size <= i && i < (f_size + fr_size)) {
        fr_outliers.push_back(i - (f_size));
      } else if ((f_size + fr_size) <= i && i < (f_size + fr_size + fl_size)) {
        fl_outliers.push_back(i - (f_size + fr_size));
      } else if ((f_size + fr_size + fl_size) <= i && i < (f_size + fr_size + fl_size + br_size)) {
        br_outliers.push_back(i - (f_size + fr_size + fl_size));
      } else if ((f_size + fr_size + fl_size + br_size) <= i) {
        bl_outliers.push_back(i - (f_size + fr_size + fl_size + br_size));
      }
    }
  }
  conti_radar::Measurement f_out, fr_out, fl_out, br_out, bl_out;
  f_out = contiMsgOutlierFilterComp(f_input, f_outliers, beta_f, calcVjxVjy(ego_mat, f_x, f_y));
  fr_out = contiMsgOutlierFilterComp(fr_input, fr_outliers, beta_fr, calcVjxVjy(ego_mat, fr_x, fr_y));
  fl_out = contiMsgOutlierFilterComp(fl_input, fl_outliers, beta_fl, calcVjxVjy(ego_mat, fl_x, fl_y));
  br_out = contiMsgOutlierFilterComp(br_input, br_outliers, beta_br, calcVjxVjy(ego_mat, br_x, br_y));
  bl_out = contiMsgOutlierFilterComp(bl_input, bl_outliers, beta_bl, calcVjxVjy(ego_mat, bl_x, bl_y));
  //-- Outliers Msg End --//

  conti_radar::Measurement f_in, fr_in, fl_in, br_in, bl_in;

  std::vector<int> f_inliers, fr_inliers, fl_inliers, br_inliers, bl_inliers;

  f_inliers = outlierToInlier(f_outliers, f_size);
  fr_inliers = outlierToInlier(fr_outliers, fr_size);
  fl_inliers = outlierToInlier(fl_outliers, fl_size);
  br_inliers = outlierToInlier(br_outliers, br_size);
  bl_inliers = outlierToInlier(bl_outliers, bl_size);

  f_in = contiMsgInlierFilter(f_input, f_inliers);
  fr_in = contiMsgInlierFilter(fr_input, fr_inliers);
  fl_in = contiMsgInlierFilter(fl_input, fl_inliers);
  br_in = contiMsgInlierFilter(br_input, br_inliers);
  bl_in = contiMsgInlierFilter(bl_input, bl_inliers);

  int out_size = f_outliers.size() + fr_outliers.size() + fl_outliers.size() +
                 br_outliers.size() + bl_outliers.size();
  int in_size = f_inliers.size() + fr_inliers.size() + fl_inliers.size() +
                br_inliers.size() + bl_inliers.size();

  if (out_size + in_size != total_size) {
    ROS_ERROR(
        "\n!!!!!!!!!!!!!!!!! \n!!!!!!!!!!!!!!!!! \n~~~~~~~~~~~~size "
        "error~~~~~~~~~~~~ \n!!!!!!!!!!!!!!!!! \n!!!!!!!!!!!!!!!!! \n");
  }
  total_size = 0;
  if (f_in.points.size() != f_inliers.size()) {
    ROS_ERROR("!!!!!! f error !!!!!!\n");
  }
  if (fr_in.points.size() != fr_inliers.size()) {
    ROS_ERROR("!!!!!! fr error !!!!!!\n");
  }
  if (fl_in.points.size() != fl_inliers.size()) {
    ROS_ERROR("!!!!!! fl error !!!!!!\n");
  }
  if (br_in.points.size() != br_inliers.size()) {
    ROS_ERROR("!!!!!! br error !!!!!!\n");
  }
  if (bl_in.points.size() != bl_inliers.size()) {
    ROS_ERROR("!!!!!! bl error !!!!!!\n");
  }

  Eigen::Matrix4f tf_td;

  common::EgoPointClouds epcs;
  sensor_msgs::PointCloud2 augpc;

  epcs.stamp = ros::Time(min_t);
  epcs.ego_vx = ego_mat(1, 0);
  epcs.ego_vy = ego_mat(2, 0);
  epcs.ego_vth = ego_mat(0, 0);

  tf_td = getTdTransform(f_t - min_t, ego_mat(1, 0), ego_mat(2, 0), ego_mat(0, 0));
  auto pcsr_f = MakePointCloudSensor("front", tf_td * tf_f, conti_to_rospc(f_in, 0), augpc);
  epcs.pcs.push_back(pcsr_f);

  tf_td = getTdTransform(fr_t - min_t, ego_mat(1, 0), ego_mat(2, 0), ego_mat(0, 0));
  auto pcsr_fr = MakePointCloudSensor("front right", tf_td * tf_fr, conti_to_rospc(fr_in, 1), augpc);
  epcs.pcs.push_back(pcsr_fr);

  tf_td = getTdTransform(fl_t - min_t, ego_mat(1, 0), ego_mat(2, 0), ego_mat(0, 0));
  auto pcsr_fl = MakePointCloudSensor("front left", tf_td * tf_fl, conti_to_rospc(fl_in, 2), augpc);
  epcs.pcs.push_back(pcsr_fl);

  tf_td = getTdTransform(br_t - min_t, ego_mat(1, 0), ego_mat(2, 0), ego_mat(0, 0));
  auto pcsr_br = MakePointCloudSensor("back right", tf_td * tf_br, conti_to_rospc(br_in, 3), augpc);
  epcs.pcs.push_back(pcsr_br);

  tf_td = getTdTransform(bl_t - min_t, ego_mat(1, 0), ego_mat(2, 0), ego_mat(0, 0));
  auto pcsr_bl = MakePointCloudSensor("back left", tf_td * tf_bl, conti_to_rospc(bl_in, 4), augpc);
  epcs.pcs.push_back(pcsr_bl);

  epcs.augpc = PointCloudAdapter(augpc);
  vepcs.push_back(epcs);

  augpc.header.stamp = ros::Time(min_t);
  augpc.header.frame_id = "/car";

  f_outlier_pub.publish(f_out);
  fr_outlier_pub.publish(fr_out);
  fl_outlier_pub.publish(fl_out);
  br_outlier_pub.publish(br_out);
  bl_outlier_pub.publish(bl_out);

  f_inlier_pub.publish(f_in);
  fr_inlier_pub.publish(fr_in);
  fl_inlier_pub.publish(fl_in);
  br_inlier_pub.publish(br_in);
  bl_inlier_pub.publish(bl_in);

  pub.publish(odom);
  pub_pc.publish(augpc);
}

void savecb(const std_msgs::Empty::ConstPtr &msg) {
  // rostopic pub /savepc std_msgs/Empty -1
  common::SerializationOutput(outfolder + "/vepcs.ser", vepcs);
  ROS_INFO("Saving... Done");
  vepcs.clear();
}

int main(int argc, char** argv) {
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("outfolder", po::value<string>(&outfolder)->required(), "EgoPointClouds folder path");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

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

  f_outlier_pub = nh.advertise<conti_radar::Measurement>("radar_front_outlier", 1);
  fr_outlier_pub = nh.advertise<conti_radar::Measurement>("radar_front_right_outlier", 1);
  fl_outlier_pub = nh.advertise<conti_radar::Measurement>("radar_front_left_outlier", 1);
  br_outlier_pub = nh.advertise<conti_radar::Measurement>("radar_back_right_outlier", 1);
  bl_outlier_pub = nh.advertise<conti_radar::Measurement>("radar_back_left_outlier", 1);

  f_inlier_pub = nh.advertise<conti_radar::Measurement>("radar_front_inlier", 1);
  fr_inlier_pub = nh.advertise<conti_radar::Measurement>("radar_front_right_inlier", 1);
  fl_inlier_pub = nh.advertise<conti_radar::Measurement>("radar_front_left_inlier", 1);
  br_inlier_pub = nh.advertise<conti_radar::Measurement>("radar_back_right_inlier", 1);
  bl_inlier_pub = nh.advertise<conti_radar::Measurement>("radar_back_left_inlier", 1);

  ros::spin();
}
