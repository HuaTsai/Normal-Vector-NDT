// Single Scene Test
#include <common/common.h>
#include <metric/metric.h>
#include <nav_msgs/Path.h>
#include <ndt/matcher.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <sndt_exec/wrapper.h>
#include <visualization_msgs/MarkerArray.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;
using Marker = visualization_msgs::Marker;

struct Res {
  void Show() {
    printf(" its: %f / %f\n", Stat(its).mean, Stat(its).max);
    printf(" err: %f / %f\n", Stat(terr).rms, Stat(rerr).rms);
    printf("corr: %f\n", Stat(corr).mean);
    double den = its.size() * 1000.;
    printf("time: %f / %f\n", timer.optimize() / den, timer.total() / den);
  }
  vector<double> its;
  vector<double> terr;
  vector<double> rerr;
  vector<double> corr;
  Timer timer;
};

Affine3d BM(const nav_msgs::Path &gt,
            const ros::Time &t1,
            const ros::Time &t2) {
  Affine3d To, Ti;
  tf2::fromMsg(GetPose(gt.poses, t2), To);
  tf2::fromMsg(GetPose(gt.poses, t1), Ti);
  return Ti.inverse() * To;
}

Marker MP(const vector<Vector3d> &points, bool red) {
  Marker ret;
  ret.header.frame_id = "map";
  ret.header.stamp = ros::Time::now();
  ret.id = 0;
  ret.type = Marker::SPHERE_LIST;
  ret.action = Marker::ADD;
  ret.scale.x = ret.scale.y = ret.scale.z = 0.5;
  ret.pose = tf2::toMsg(Affine3d::Identity());
  ret.color.a = 1;
  ret.color.r = red ? 1 : 0;
  ret.color.g = red ? 0 : 1;
  for (const auto &point : points) {
    geometry_msgs::Point pt;
    pt.x = point(0), pt.y = point(1), pt.z = point(2);
    ret.points.push_back(pt);
  }
  return ret;
}

visualization_msgs::Marker Boxes(std::shared_ptr<NMap> map, bool red) {
  visualization_msgs::Marker ret;
  ret.header.stamp = ros::Time::now();
  ret.header.frame_id = "map";
  ret.type = visualization_msgs::Marker::CUBE_LIST;
  geometry_msgs::Vector3 s;
  double cs = map->GetCellSize();
  s.x = cs, s.y = cs, s.z = cs;
  ret.scale = s;
  ret.pose.orientation.w = 1;
  for (auto [idx, cell] : *map) {
    static_cast<void>(idx);
    auto cen = cell.GetCenter();
    geometry_msgs::Point c;
    c.x = cen(0), c.y = cen(1), c.z = cen(2);
    ret.points.push_back(c);
    std_msgs::ColorRGBA cl;
    cl.a = 0.2, cl.r = red ? 1 : 0, cl.g = red ? 0 : 1;
    ret.colors.push_back(cl);
  }
  return ret;
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  string d;
  bool tr;
  int n, f;
  double ndtd2, nndtd2, cs;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
      ("n,n", po::value<int>(&n)->default_value(0), "N")
      ("f,f", po::value<int>(&f)->default_value(1), "F")
      ("ndtd2,a", po::value<double>(&ndtd2)->default_value(0.3), "ndtd2")
      ("nndtd2,b", po::value<double>(&nndtd2)->default_value(0.3), "nndtd2")
      ("cs,c", po::value<double>(&cs)->default_value(1), "cell size")
      ("tr", po::value<bool>(&tr)->default_value(false)->implicit_value(true), "Trust Region");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  nav_msgs::Path gt;
  SerializationInput(JoinPath(GetDataPath(d), "gt.ser"), gt);
  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(d), "lidar.ser"), vpc);

  std::vector<double> bent, benr;
  Res r1, r2;
  auto tgt = PCMsgTo3D(vpc[n], 0);
  auto src = PCMsgTo3D(vpc[n + f], 0);
  TransformPointsInPlace(tgt, aff3);
  TransformPointsInPlace(src, aff3);
  auto ben = BM(gt, vpc[n].header.stamp, vpc[n + f].header.stamp);
  cout << "Bench Mark: " << TransNormRotDegAbsFromAffine3d(ben).transpose()
       << endl;

  NDTMatcher m1({tr ? kTR : kLS, kNDT, k1to1}, cs, ndtd2);
  m1.SetSource(src);
  m1.SetTarget(tgt);
  auto res1 = m1.Align();
  auto err1 = TransNormRotDegAbsFromAffine3d(res1 * ben.inverse());
  r1.terr.push_back(err1(0));
  r1.rerr.push_back(err1(1));
  r1.corr.push_back(m1.corres());
  r1.timer += m1.timer();
  r1.its.push_back(m1.iteration());

  NDTMatcher m2({tr ? kTR : kLS, kNNDT, k1to1}, cs, nndtd2);
  m2.SetSource(src);
  m2.SetTarget(tgt);
  auto res2 = m2.Align();
  auto err2 = TransNormRotDegAbsFromAffine3d(res2 * ben.inverse());
  r2.terr.push_back(err2(0));
  r2.rerr.push_back(err2(1));
  r2.corr.push_back(m2.corres());
  r2.timer += m2.timer();
  r2.its.push_back(m2.iteration());

  r1.Show();
  cout << "-------------" << endl;
  r2.Show();

  ros::init(argc, argv, "exp7_2");
  ros::NodeHandle nh;
  ros::Publisher pub1 = nh.advertise<Marker>("src", 0, true);
  ros::Publisher pub2 = nh.advertise<Marker>("tgt", 0, true);
  ros::Publisher pub3 = nh.advertise<Marker>("out1", 0, true);
  ros::Publisher pub4 = nh.advertise<Marker>("out2", 0, true);
  ros::Publisher pub5 = nh.advertise<Marker>("box1", 0, true);
  ros::Publisher pub6 = nh.advertise<Marker>("box2", 0, true);
  ros::Publisher pub7 = nh.advertise<Marker>("box3", 0, true);
  pub1.publish(MP(src, false));
  pub2.publish(MP(tgt, true));
  auto out1 = TransformPoints(src, res1);
  auto out2 = TransformPoints(src, res2);
  pub3.publish(MP(out1, false));
  pub4.publish(MP(out2, false));
  pub5.publish(Boxes(m1.tmap(), true));

  auto mp1 = std::make_shared<NMap>(cs);
  mp1->LoadPoints(out1);
  pub6.publish(Boxes(mp1, false));

  auto mp2 = std::make_shared<NMap>(cs);
  mp2->LoadPoints(out2);
  pub7.publish(Boxes(mp2, false));
  ros::spin();
}
