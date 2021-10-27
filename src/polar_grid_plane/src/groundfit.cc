/*
    @file groundplanfit.cpp
    @brief ROS Node for ground plane fitting
    This is a ROS node to perform ground plan fitting.
    Implementation accoriding to <Fast Segmentation of 3D Point Clouds: A
   Paradigm> In this case, it's assumed that the x,y axis points at sea-level,
    and z-axis points up. The sort of height is based on the Z-axis value.
    @author Vincent Cheung(VincentCheungm)
    @bug Sometimes the plane is not fit.
*/
/* This file is modified to offline by HuaTsai */

#include <iostream>
#include <list>
#include <vector>
// For disable PCL complile lib, to use PointXYZIR
#define PCL_NO_PRECOMPILE

#include <common/common.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <omp.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/point_cloud.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

// Customed Point Struct for holding clustered points
namespace scan_line_run {
/** Euclidean Velodyne coordinate, including intensity and ring number, and
 * label. */
struct PointXYZIRL {
  PCL_ADD_POINT4D;                 // quad-word XYZ
  float intensity;                 ///< laser intensity reading
  uint16_t ring;                   ///< laser ring number
  uint16_t label;                  ///< point label
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment
} EIGEN_ALIGN16;

};  // namespace scan_line_run

#define SLRPointXYZIRL scan_line_run::PointXYZIRL
#define RUN pcl::PointCloud<SLRPointXYZIRL>
// Register custom point struct according to PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(
    scan_line_run::PointXYZIRL,
    (float, x, x)(float, y, y)(float, z, z)(float,
                                            intensity,
                                            intensity)(uint16_t,
                                                       ring,
                                                       ring)(uint16_t,
                                                             label,
                                                             label))

// using eigen lib
#include <Eigen/Dense>
using Eigen::JacobiSVD;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using namespace std;

int hori_seg, vert_seg, num_iter;
double vert_dist;
pcl::PointCloud<pcl::PointXYZL>::Ptr g_seeds_pc(
    new pcl::PointCloud<pcl::PointXYZL>());
pcl::PointCloud<pcl::PointXYZL>::Ptr g_all_seeds_pc(
    new pcl::PointCloud<pcl::PointXYZL>());
pcl::PointCloud<pcl::PointXYZL>::Ptr segmented_ground_pc(
    new pcl::PointCloud<pcl::PointXYZL>());
pcl::PointCloud<pcl::PointXYZL>::Ptr segmented_not_ground_pc(
    new pcl::PointCloud<pcl::PointXYZL>());
pcl::PointCloud<SLRPointXYZIRL>::Ptr g_all_pc(
    new pcl::PointCloud<SLRPointXYZIRL>());

/*
    @brief Compare function to sort points. Here use z axis.
    @return z-axis accent
*/
bool point_cmp(pcl::PointXYZL a, pcl::PointXYZL b) { return a.z < b.z; }

/*
    @brief Ground Plane fitting ROS Node.
    @param Velodyne Pointcloud topic.
    @param Sensor Model.
    @param Sensor height for filtering error mirror points.
    @param Num of segment, iteration, LPR
    @param Threshold of seeds distance, and ground plane distance

    @subscirbe:/velodyne_points
    @publish:/points_no_ground, /points_ground
*/

class PointContainer {
 public:
  inline bool IsEmpty() const { return _points.empty(); }
  inline std::vector<size_t>& points() { return _points; }
  inline const std::vector<size_t>& points() const { return _points; }

 private:
  std::vector<size_t> _points;
};

class GroundPlaneFit {
  using PointColumn = std::vector<PointContainer>;
  using PointMatrix = std::vector<PointColumn>;

 public:
  GroundPlaneFit();
  sensor_msgs::PointCloud2 velodyne_callback_(
      const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);

 private:
  int sensor_model_;
  double sensor_height_;
  int num_seg_;
  int num_iter_;
  int num_lpr_;
  int m;
  int n;
  int seg_m;
  int seg_n;
  int cube_counter;
  double th_seeds_;
  double th_dist_;
  bool reasonable_normal;

  void estimate_plane(int);
  void horizontal_seg(pcl::PointCloud<pcl::PointXYZI>,
                      vector<vector<PointContainer>>& data);
  void vertical_seg(void);
  void extract_ground(pcl::PointCloud<pcl::PointXYZL>);
  void extract_initial_seeds(const pcl::PointCloud<pcl::PointXYZL>& p_sorted);
  void remove_ego_points(
      const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
      pcl::PointCloud<pcl::PointXYZI>&,
      pcl::PointCloud<pcl::PointXYZI>&);

  // Model parameter for ground plane fittings
  // The ground plane model is: ax+by+cz+d=0
  // Here normal:=[a,b,c], d=d
  // th_dist_d_ = threshold_dist - d
  float d_;
  MatrixXf normal_;
  float th_dist_d_;

  vector<int> ground_index;
};

/*
    @brief Constructor of GPF Node.
    @return void
*/
GroundPlaneFit::GroundPlaneFit() {
  printf("Inititalizing Ground Plane Fitter...\n");
  sensor_model_ = 32;
  printf("Sensor Model: %d\n", sensor_model_);
  sensor_height_ = 2.5;
  printf("Sensor Height: %f\n", sensor_height_);
  num_seg_ = 3;
  printf("Num of Segments: %d\n", num_seg_);
  num_iter_ = num_iter;
  printf("Num of Iteration: %d\n", num_iter_);
  num_lpr_ = 20;
  printf("Num of LPR: %d\n", num_lpr_);
  th_seeds_ = 0.4;
  printf("Seeds Threshold: %f\n", th_seeds_);
  th_dist_ = 0.2;
  printf("Distance Threshold: %f\n", th_dist_);
  m = hori_seg;
  n = vert_seg;
}

/*
    @brief The function to estimate plane model. The
    model parameter `normal_` and `d_`, and `th_dist_d_`
    is set here.
    The main step is performed SVD(UAV) on covariance matrix.
    Taking the sigular vector in U matrix according to the smallest
    sigular value in A, as the `normal_`. `d_` is then calculated
    according to mean ground points.
    @param g_ground_pc:global ground pointcloud ptr.

*/
void GroundPlaneFit::estimate_plane(int num_iter) {
  // Create covarian matrix in single pass.
  Eigen::Matrix3f cov;
  Eigen::Vector4f pc_mean;
  if (segmented_ground_pc->points.size() >= 3) {
    pcl::computeMeanAndCovarianceMatrix(*segmented_ground_pc, cov, pc_mean);
    // Singular Value Decomposition: SVD
    JacobiSVD<MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));
    // mean ground seeds value
    Eigen::Vector3f seeds_mean = pc_mean.head<3>();

    Eigen::Vector3f v1(normal_(0, 0), normal_(1, 0), normal_(2, 0));
    Eigen::Vector3f v2(0.0, 0.0, 1.0);

    double angle_diff = acos(v1.dot(v2));

    if (angle_diff <
        0.35)  // angle_diff < 60 degree (1.0471975512) 40(0.69813170079)
      reasonable_normal = true;
    else
      reasonable_normal = false;

    Eigen::Quaternionf out;
    out.setFromTwoVectors(v1, v2);

    // according to normal.T*[x,y,z] = -d
    d_ = (normal_.transpose() * seeds_mean)(0, 0);
    // set distance threhold to `th_dist - d`
    th_dist_d_ = th_dist_ - d_;

    cube_counter++;

  } else {
    // cout<<"No enough points for plane extraction"<<endl;
    return;
  }
}

/*
    @brief Extract initial seeds of the given pointcloud sorted segment.
    This function filter ground seeds points accoring to heigt.
    This function will set the `g_ground_pc` to `g_seed_pc`.
    @param p_sorted: sorted pointcloud

    @param ::num_lpr_: num of LPR points
    @param ::th_seeds_: threshold distance of seeds
    @param ::
*/
void GroundPlaneFit::extract_initial_seeds(
    const pcl::PointCloud<pcl::PointXYZL>& p_sorted) {
  // LPR is the mean of low point representative
  double sum = 0;
  int cnt = 0;
  // Calculate the mean height value.
  for (size_t i = 0; i < p_sorted.points.size() && cnt < num_lpr_; i++) {
    if (p_sorted.points[i].z > -3.5) {
      sum += p_sorted.points[i].z;
      cnt++;
    }
  }
  double lpr_height = cnt != 0 ? sum / cnt : 0;  // in case divide by 0
  g_seeds_pc->clear();
  for (size_t i = 0; i < p_sorted.points.size(); i++) {
    if ((p_sorted.points[i].z < lpr_height + th_seeds_) &&
        (p_sorted.points[i].z > lpr_height - th_seeds_)) {
      g_seeds_pc->points.push_back(p_sorted.points[i]);
    }
  }
}

/*
    @brief Velodyne pointcloud callback function. The main GPF pipeline is here.
    PointCloud SensorMsg -> Pointcloud -> z-value sorted Pointcloud
    ->error points removal -> extract ground seeds -> ground plane fit mainloop
*/
void GroundPlaneFit::horizontal_seg(
    pcl::PointCloud<pcl::PointXYZI> laserCloudIn,
    vector<vector<PointContainer>>& data) {
  double angle_div = 2 * M_PI / m;

  for (size_t i = 0; i < laserCloudIn.points.size(); i++) {
    const auto& point = laserCloudIn.points[i];
    double angle = atan2(point.y, point.x) + M_PI;
    double dist = sqrt(point.x * point.x + point.y * point.y);
    int hor_seg = (floor)(angle / angle_div);
    int vert_seg = (floor)(dist / vert_dist);

    if (hor_seg > m - 1 || hor_seg < 0) {
      data[0][vert_seg].points().push_back(i);
      continue;
    }

    if (vert_seg >= n - 1) {
      cout << "radial direction larger than expected: " << dist << endl;
      cout << "limited: " << vert_dist * (n) << endl;
      continue;
    }

    size_t index = i;
    data[hor_seg][vert_seg].points().push_back(index);
  }
}

void GroundPlaneFit::extract_ground(
    pcl::PointCloud<pcl::PointXYZL> segmented_cloud) {
  segmented_ground_pc->clear();
  segmented_not_ground_pc->clear();
  extract_initial_seeds(segmented_cloud);
  segmented_ground_pc = g_seeds_pc;

  MatrixXf points;
  points.setZero(segmented_cloud.points.size(), 3);
  int j = 0;

  for (auto p : segmented_cloud.points) {
    points.row(j++) << p.x, p.y, p.z;
  }

  for (int i = 0; i < num_iter_; i++) {
    estimate_plane(i);
    segmented_ground_pc->clear();
    segmented_not_ground_pc->clear();

    VectorXf result = points * normal_;

    for (int r = 0; r < result.rows(); r++) {
      int index = segmented_cloud.points[r].label;

      if (result[r] > d_ - th_dist_ && result[r] < d_ + th_dist_) {
        if (i == num_iter_ - 1) {
          if (reasonable_normal)
            g_all_pc->points[index].label = 0u;
          else
            g_all_pc->points[index].label = 1u;
        }

        segmented_ground_pc->points.push_back(segmented_cloud[r]);
      } else {
        if (i == num_iter_ - 1) g_all_pc->points[index].label = 1u;

        segmented_not_ground_pc->points.push_back(segmented_cloud[r]);
      }
    }
  }
}

void GroundPlaneFit::remove_ego_points(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr,
    pcl::PointCloud<pcl::PointXYZI>& out_filtered_cloud,
    pcl::PointCloud<pcl::PointXYZI>& ego_cloud) {
  pcl::ExtractIndices<pcl::PointXYZI> extractor;
  extractor.setInputCloud(in_cloud_ptr);
  pcl::PointIndices indices;

  for (size_t i = 0; i < in_cloud_ptr->points.size(); i++) {
    pcl::PointXYZI temp_point;
    temp_point = in_cloud_ptr->points[i];

    if ((temp_point.x * temp_point.x + temp_point.y * temp_point.y +
         temp_point.z * temp_point.z) < 9)
      indices.indices.push_back(i);
  }
  extractor.setIndices(boost::make_shared<pcl::PointIndices>(indices));
  extractor.setNegative(true);
  extractor.filter(out_filtered_cloud);
  extractor.setNegative(false);
  extractor.filter(ego_cloud);
}

sensor_msgs::PointCloud2 GroundPlaneFit::velodyne_callback_(
    const sensor_msgs::PointCloud2ConstPtr& in_cloud_msg) {
  // 1.Msg to pointcloud
  pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudIn_org(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*in_cloud_msg, *laserCloudIn_org);

  pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
  pcl::PointCloud<pcl::PointXYZI> ego_cloud;

  remove_ego_points(laserCloudIn_org, laserCloudIn, ego_cloud);

  cube_counter = 0;
  // For mark ground points and hold all points
  SLRPointXYZIRL point;
  for (size_t i = 0; i < laserCloudIn.points.size(); i++) {
    laserCloudIn.points[i].z = laserCloudIn.points[i].z;
    point.x = laserCloudIn.points[i].x;
    point.y = laserCloudIn.points[i].y;
    point.z = laserCloudIn.points[i].z;
    point.intensity = laserCloudIn.points[i].intensity;
    // point.ring = laserCloudIn.points[i].ring;
    point.label = 0u;  // 0 means uncluster
    g_all_pc->points.push_back(point);
  }

  vector<PointColumn> _data(m, vector<PointContainer>(n));
  horizontal_seg(laserCloudIn, _data);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      pcl::PointCloud<pcl::PointXYZL> segmented_cloud;
      vector<size_t> points_index(_data[i][j].points().size(), 0);

      segmented_cloud.clear();

      for (size_t k = 0; k < _data[i][j].points().size(); k++) {
        size_t index = _data[i][j].points()[k];
        const auto& point = laserCloudIn.points[index];

        pcl::PointXYZL current_point;

        current_point.x = point.x;
        current_point.y = point.y;
        current_point.z = point.z;
        current_point.label = index;

        segmented_cloud.points.push_back(current_point);
      }

      sort(segmented_cloud.points.begin(), segmented_cloud.end(), point_cmp);

      seg_m = i;
      seg_n = j;
      if (segmented_cloud.points.size() >= 3)
        extract_ground(segmented_cloud);
      else
        continue;
    }
  }

  segmented_ground_pc->clear();
  segmented_not_ground_pc->clear();

  pcl::PointCloud<pcl::PointXYZI> nonground_seg;

  for (size_t i = 0; i < g_all_pc->points.size(); i++) {
    int label = g_all_pc->points[i].label;

    pcl::PointXYZI temp_point;

    temp_point.x = g_all_pc->points[i].x;
    temp_point.y = g_all_pc->points[i].y;
    temp_point.z = g_all_pc->points[i].z;
    temp_point.intensity = g_all_pc->points[i].intensity;

    if (!(label == 0u && g_all_pc->points[i].z > -3.5))
      nonground_seg.points.push_back(temp_point);
  }
  g_all_pc->clear();
  sensor_msgs::PointCloud2 ret;
  pcl::toROSMsg(nonground_seg, ret);
  ret.header = in_cloud_msg->header;
  return ret;
}

int main(int argc, char** argv) {
  std::string data, topic;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("topic", po::value<string>(&topic)->default_value("nuscenes"), "Topic")
      ("hori_seg", po::value<int>(&hori_seg)->default_value(15), "Horizontal Segments")
      ("vert_seg", po::value<int>(&vert_seg)->default_value(20), "Vertical Segments")
      ("vert_dist", po::value<double>(&vert_dist)->default_value(7.), "Vertical Distance")
      ("num_iter", po::value<int>(&num_iter)->default_value(3), "Number of Iterations");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  std::cout << "sub_topic: " << topic << std::endl;
  std::cout << "horizontal: " << hori_seg
            << " angle division: " << 2 * M_PI / hori_seg * 180 / M_PI << endl;
  std::cout << "vertical: " << vert_seg
            << "  max radial distance: " << vert_seg * vert_dist << endl;

  vector<sensor_msgs::PointCloud2ConstPtr> pcs;
  for (auto path : GetBagsPath(data)) {
    rosbag::Bag bag;
    bag.open(path);
    for (rosbag::MessageInstance const m : rosbag::View(bag)) {
      sensor_msgs::PointCloud2ConstPtr msg =
          m.instantiate<sensor_msgs::PointCloud2>();
      if (msg && m.getTopic() == "nuscenes_lidar") pcs.push_back(msg);
    }
    bag.close();
  }
  GroundPlaneFit node;
  vector<sensor_msgs::PointCloud2> pcs_seg;
  int n = pcs.size();
  for (int i = 0; i < n; ++i) {
    printf("\rGround Segentation: (%d/%d)", i, n - 1);
    fflush(stdout);
    auto ng = node.velodyne_callback_(pcs[i]);
    pcs_seg.push_back(ng);
  }
  printf("\n");
  SerializationOutput(JoinPath(GetDataPath(data), "lidar.ser"), pcs_seg);
}