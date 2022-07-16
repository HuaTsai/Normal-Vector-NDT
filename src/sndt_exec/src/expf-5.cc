// nuScenes, NDT, different bags, visualization

#include <common/common.h>
#include <metric/metric.h>
#include <nav_msgs/Path.h>
#include <ndt/matcher.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <polar_grid_plane/groundfit.h>
#include <rosbag/view.h>
#include <sndt_exec/wrapper.h>
#include <tf2_msgs/TFMessage.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>

using namespace std;

using namespace Eigen;
using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;
namespace po = boost::program_options;

void RunMatch(const vector<Vector3d> &src,
              const vector<Vector3d> &tgt,
              NDTMatcher &m) {
  m.set_intrinsic(0.00005);
  m.SetSource(src);
  m.SetTarget(tgt);
  m.Align();
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  double d2 = 0.05;
  double cs;
  int sceneid;
  int id, df;

  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<int>(&sceneid)->required(), "scene id")
      ("i,i", po::value<int>(&id)->required(), "Index")
      ("f,f", po::value<int>(&df)->required(), "Diff")
      ("c,c", po::value<double>(&cs)->required(), "Cell Size");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  GroundPlaneFit gpf;

  // Fetch Data
  rosbag::Bag bag;
  bag.open(GetScenePath(sceneid));
  vector<vector<Vector3d>> vpc;
  vector<ros::Time> ts;
  vector<geometry_msgs::PoseStamped> psts;
  for (rosbag::MessageInstance const m : rosbag::View(bag)) {
    sensor_msgs::PointCloud2ConstPtr pcmsg =
        m.instantiate<sensor_msgs::PointCloud2>();
    if (pcmsg && m.getTopic() == "nuscenes_lidar") {
      auto pc = gpf.velodyne_callback_(pcmsg);
      vpc.push_back(TransformPoints(PCMsgTo3D(pc, 0), aff3));
      ts.push_back(pcmsg->header.stamp);
    }

    tf2_msgs::TFMessageConstPtr tfmsg = m.instantiate<tf2_msgs::TFMessage>();
    if (tfmsg) {
      auto tf = tfmsg->transforms.at(0);
      if (tf.header.frame_id == "map" && tf.child_frame_id == "car") {
        geometry_msgs::PoseStamped pst;
        pst.header = tf.header;
        pst.pose.position.x = tf.transform.translation.x;
        pst.pose.position.y = tf.transform.translation.y;
        pst.pose.position.z = tf.transform.translation.z;
        pst.pose.orientation = tf.transform.rotation;
        psts.push_back(pst);
      }
    }
  }
  bag.close();

  // -d 429 -c 1 -i 338 -f 1
  // Match Pin
  if (id == -1) {
    for (size_t x = 0; x < vpc.size() - 1u; ++x) {
      auto tgt = vpc[x];
      auto src = vpc[x + df];

      unordered_set<Options> op1 = {kNDT, k1to1, kPointCov, kAnalytic,
                                    kNoReject};
      auto m1 = NDTMatcher::GetBasic(op1, cs, d2);
      RunMatch(src, tgt, m1);

      unordered_set<Options> op2 = {kNNDT, k1to1, kPointCov, kAnalytic,
                                    kNoReject};
      auto m2 = NDTMatcher::GetBasic(op2, cs, d2);
      RunMatch(src, tgt, m2);

      if (m1.iteration() - m2.iteration() >= 2)
        cout << x << " -> " << m1.iteration() - m2.iteration() << endl;
    }
  } else {
    ros::init(argc, argv, "expf_5");
    ros::NodeHandle nh;
    auto pub1 = nh.advertise<sensor_msgs::PointCloud2>("pc1", 0, true);
    auto pub2 = nh.advertise<sensor_msgs::PointCloud2>("pc2", 0, true);
    auto pub3 = nh.advertise<sensor_msgs::PointCloud2>("pc3", 0, true);

    auto tgt = vpc[id];
    auto src = vpc[id + df];

    PointCloudType::Ptr source_pcl = PointCloudType::Ptr(new PointCloudType);
    PointCloudType::Ptr target_pcl = PointCloudType::Ptr(new PointCloudType);
    for (const auto &pt : src) {
      pcl::PointXYZ p;
      p.x = pt(0), p.y = pt(1), p.z = pt(2);
      source_pcl->push_back(p);
    }
    for (const auto &pt : tgt) {
      pcl::PointXYZ p;
      p.x = pt(0), p.y = pt(1), p.z = pt(2);
      target_pcl->push_back(p);
    }
    source_pcl->header.frame_id = "map";
    target_pcl->header.frame_id = "map";
    pub1.publish(*target_pcl);

    unordered_set<Options> op1 = {kNDT, k1to1, kPointCov, kAnalytic, kNoReject};
    auto m1 = NDTMatcher::GetBasic(op1, cs, d2);
    RunMatch(src, tgt, m1);

    unordered_set<Options> op2 = {kNNDT, k1to1, kPointCov, kAnalytic,
                                  kNoReject};
    auto m2 = NDTMatcher::GetBasic(op2, cs, d2);
    RunMatch(src, tgt, m2);

    cout << m1.iteration() << " vs. " << m2.iteration() << endl;

    PointCloudType o1, o2;
    size_t it;
    cout << "Iter: ";
    while (cin >> it) {
      PointCloudType o1, o2;
      if (it < m1.tfs().size()) {
        pcl::transformPointCloud(*source_pcl, o1, m1.tfs()[it].cast<float>());
        pub2.publish(o1);
      }

      if (it < m2.tfs().size()) {
        pcl::transformPointCloud(*source_pcl, o2, m2.tfs()[it].cast<float>());
        pub3.publish(o2);
      }
      ros::spinOnce();
      cout << "Iter: ";
    }
  }
}
