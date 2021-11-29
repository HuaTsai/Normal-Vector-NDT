// Different Scans
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <std_msgs/Int32.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.h>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;
const auto &Avg = Average;

vector<vector<visualization_msgs::MarkerArray>> ms;
ros::Publisher pub1, pub2, pub3, pub4;

void Q1MedianQ3(vector<double> &data) {
  sort(data.begin(), data.end());
  printf("[%g, %g, %g, %g, %g]\n", data[0], data[data.size() / 4],
         data[data.size() / 2], data[data.size() / 4 * 3],
         data[data.size() - 1]);
}

double RMS(const vector<Vector2d> &tgt,
           const vector<Vector2d> &src,
           const Affine2d &aff = Affine2d::Identity()) {
  vector<Vector2d> src2(src.size());
  transform(src.begin(), src.end(), src2.begin(),
            [&aff](auto p) { return aff * p; });
  double ret = 0;
  for (size_t i = 0; i < src2.size(); ++i)
    ret += (src2[i] - tgt[i]).squaredNorm();
  return sqrt(ret / src2.size());
}

void OutlierRemove(vector<Vector2d> &data) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto pt : data) pc->push_back(pcl::PointXYZ(pt(0), pt(1), 0));
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> filter(true);
  filter.setInputCloud(pc);
  filter.setMeanK(8);
  filter.setStddevMulThresh(1.0);
  vector<int> indices;
  filter.filter(indices);
  data.clear();
  for (auto i : indices) data.push_back(Vector2d(pc->at(i).x, pc->at(i).y));
}

Affine2d GetGtPose(string data, ros::Time tt, ros::Time ts) {
  nav_msgs::Path gtpath;
  string base = "/home/ee904/Desktop/HuaTsai/NormalNDT/Analysis/1Data/" + data;
  SerializationInput(base + "/gt.ser", gtpath);
  Affine3d Tt, Ts;
  tf2::fromMsg(GetPose(gtpath.poses, tt), Tt);
  tf2::fromMsg(GetPose(gtpath.poses, ts), Ts);
  Affine3d Tts = Conserve2DFromAffine3d(Tt.inverse() * Ts);
  Affine2d ret = Translation2d(Tts.translation()(0), Tts.translation()(1)) *
                 Rotation2Dd(Tts.rotation().block<2, 2>(0, 0));
  cout << "td: " << (ts - tt).toSec() << ", ";
  cout << "(r, t) = " << TransNormRotDegAbsFromAffine2d(ret).transpose()
       << endl;
  return ret;
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n, m, o;
  double cell_size, huber, voxel, x, y, t, r, radius;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("huber,u", po::value<double>(&huber)->default_value(1.0), "Use Huber loss")
      ("radius,a", po::value<double>(&radius)->default_value(1.5), "Use Huber loss")
      ("n,n", po::value<int>(&n)->default_value(0), "n")
      ("x,x", po::value<double>(&x)->default_value(0), "x")
      ("y,y", po::value<double>(&y)->default_value(0), "y")
      ("t,t", po::value<double>(&t)->default_value(0), "t")
      ("r,r", po::value<double>(&r)->default_value(15), "r")
      ("m,m", po::value<int>(&m)->default_value(1), "r")
      ("o,o", po::value<int>(&o)->default_value(1), "offset");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  ros::init(argc, argv, "evalsndt");
  ros::NodeHandle nh;
  pub1 = nh.advertise<MarkerArray>("markers1", 0, true);
  pub2 = nh.advertise<MarkerArray>("markers2", 0, true);
  pub3 = nh.advertise<MarkerArray>("markers3", 0, true);
  pub4 = nh.advertise<MarkerArray>("markers4", 0, true);

  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);

  auto tgt = PCMsgTo2D(vpc[n], voxel);
  auto src = PCMsgTo2D(vpc[n + o], voxel);
  transform(tgt.begin(), tgt.end(), tgt.begin(),
            [&aff2](auto p) { return aff2 * p; });
  transform(src.begin(), src.end(), src.begin(),
            [&aff2](auto p) { return aff2 * p; });
  auto affgt = GetGtPose(data, vpc[n].header.stamp, vpc[n + o].header.stamp);
  transform(src.begin(), src.end(), src.begin(),
            [&affgt](auto p) { return affgt * p; });
  // OutlierRemove(tgt);
  // OutlierRemove(src);
  pub1.publish(JoinMarkers({MarkerOfPoints(tgt, 0.5, Color::kRed)}));
  pub4.publish(JoinMarkers({MarkerOfPoints(src)}));

  int samples = 15;
  auto affs = RandomTransformGenerator2D(r).Generate(samples);
  cout << "r = " << r << endl;
  for (auto aff : affs) {
    vector<Vector2d> srcc(src.size());
    transform(src.begin(), src.end(), srcc.begin(),
              [&aff](auto p) { return aff * p; });
    vector<pair<vector<Vector2d>, Affine2d>> datat{
        {tgt, Eigen::Affine2d::Identity()}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{
        {srcc, Eigen::Affine2d::Identity()}};

    CommonParameters *params;
    Affine2d T;
    ICPParameters params1;
    Pt2plICPParameters params2;
    SICPParameters params3;
    P2DNDTParameters params4;
    D2DNDTParameters params5;
    SNDTParameters params6;
    D2DNDTParameters params7;
    if (m == 1) {
      params = &params1;
      auto tgt1 = MakePoints(datat, params1);
      auto src1 = MakePoints(datas, params1);
      T = ICPMatch(tgt1, src1, params1);
    } else if (m == 2) {
      params = &params2;
      params2.radius = radius;
      auto tgt2 = MakePoints(datat, params2);
      auto src2 = MakePoints(datas, params2);
      T = Pt2plICPMatch(tgt2, src2, params2);
    } else if (m == 3) {
      params = &params3;
      params3.radius = radius;
      auto tgt3 = MakePoints(datat, params3);
      auto src3 = MakePoints(datas, params3);
      T = SICPMatch(tgt3, src3, params3);
    } else if (m == 4) {
      params = &params4;
      params4.cell_size = cell_size;
      params4.r_variance = params4.t_variance = 0;
      auto tgt4 = MakeNDTMap(datat, params4);
      auto src4 = MakePoints(datas, params4);
      T = P2DNDTMDMatch(tgt4, src4, params4);
    } else if (m == 5) {
      params = &params5;
      params5.cell_size = cell_size;
      params5.r_variance = params5.t_variance = 0;
      auto tgt5 = MakeNDTMap(datat, params5);
      auto src5 = MakeNDTMap(datas, params5);
      T = D2DNDTMDMatch(tgt5, src5, params5);
    } else if (m == 6) {
      params = &params6;
      params6.cell_size = cell_size;
      params6.radius = radius;
      params6.r_variance = params6.t_variance = 0;
      auto tgt6 = MakeSNDTMap(datat, params6);
      auto src6 = MakeSNDTMap(datas, params6);
      T = SNDTMDMatch(tgt6, src6, params6);
      pub3.publish(MarkerArrayOfSNDTMap(tgt6, true));
    } else if (m == 7) {
      params = &params7;
      params7.cell_size = cell_size;
      params7.r_variance = params7.t_variance = 0;
      auto tgt7 = MakeNDTMap(datat, params7);
      auto src7 = MakeNDTMap(datas, params7);
      T = SNDTMDMatch2(tgt7, src7, params7);
    }

    if ((aff * T).translation().isZero(1)) {
      cout << "s: " << TransNormRotDegAbsFromAffine2d(aff * T).transpose()
           << ", ";
    } else {
      cout << "f: " << TransNormRotDegAbsFromAffine2d(aff * T).transpose()
           << ", ";
    }
    cout << "Iter: " << params->_iteration << " & " << params->_ceres_iteration;
    cout << ", " << int(params->_converge) << endl;
    for (auto tf : params->_sols) {
      auto src2 = TransformPoints(srcc, tf.front());
      pub2.publish(JoinMarkers({MarkerOfPoints(src2)}));
      ros::Rate(10).sleep();
    }
  }
}
