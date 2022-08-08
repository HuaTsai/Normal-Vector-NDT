// Single Scene Test
#include <common/common.h>
#include <metric/metric.h>
#include <nav_msgs/Path.h>
#include <ndt/matcher.h>
#include <ndt/visuals.h>
#include <ndt_exec/wrapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;
namespace po = boost::program_options;

struct Res {
  void Show() {
    printf(" its: %d\n", it);
    printf(" err: %f / %f\n", terr, rerr);
    printf("corr: %d\n", corr);
    double den = 1000.;
    printf("time: %f / %f\n", timer.optimize() / den, timer.total() / den);
  }
  int it, corr;
  double terr, rerr;
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
    if (!cell.GetHasGaussian()) continue;
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
  int n, f;
  double ndtd2, nndtd2;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
      ("n,n", po::value<int>(&n)->default_value(0), "N")
      ("f,f", po::value<int>(&f)->default_value(1), "F")
      ("ndtd2,a", po::value<double>(&ndtd2)->default_value(0.05), "ndtd2")
      ("nndtd2,b", po::value<double>(&nndtd2)->default_value(0.05), "nndtd2");
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
  cout << src.size() << " -> " << tgt.size() << endl;
  TransformPointsInPlace(tgt, aff3);
  TransformPointsInPlace(src, aff3);
  auto ben = BM(gt, vpc[n].header.stamp, vpc[n + f].header.stamp);
  cout << "Bench Mark: " << TransNormRotDegAbsFromAffine3d(ben).transpose()
       << endl;

  cout << "NDT" << endl;
  auto op1 = {kNDT, k1to1, kPointCov, kAnalytic};
  auto m1 = NDTMatcher::GetBasic(op1, 0.5, ndtd2);
  m1.set_intrinsic(0.005);
  m1.SetSource(src);
  m1.SetTarget(tgt);
  auto res1 = m1.Align();
  auto err1 = TransNormRotDegAbsFromAffine3d(res1 * ben.inverse());
  r1.terr = err1(0), r1.rerr = err1(1);
  r1.corr = m1.corres(), r1.it = m1.iteration();
  r1.timer += m1.timer();

  cout << "NNDT" << endl;
  auto op2 = {kNNDT, k1to1, kPointCov, kAnalytic};
  auto m2 = NDTMatcher::GetBasic(op2, 0.5, nndtd2);
  m2.set_intrinsic(0.005);
  m2.SetSource(src);
  m2.SetTarget(tgt);
  auto res2 = m2.Align();
  auto err2 = TransNormRotDegAbsFromAffine3d(res2 * ben.inverse());
  r2.terr = err2(0), r2.rerr = err2(1);
  r2.corr = m2.corres(), r2.it = m2.iteration();
  r2.timer += m2.timer();

  r1.Show();
  cout << "-------------" << endl;
  r2.Show();

  ros::init(argc, argv, "exp7_2");
  ros::NodeHandle nh;
  ros::Publisher pu1 = nh.advertise<Marker>("marker1", 0, true);
  ros::Publisher pu2 = nh.advertise<Marker>("marker2", 0, true);
  ros::Publisher pu3 = nh.advertise<Marker>("marker3", 0, true);
  ros::Publisher pu4 = nh.advertise<Marker>("marker4", 0, true);
  ros::Publisher pub1 = nh.advertise<MarkerArray>("markers1", 0, true);
  ros::Publisher pub2 = nh.advertise<MarkerArray>("markers2", 0, true);
  ros::Publisher pub3 = nh.advertise<MarkerArray>("markers3", 0, true);

  pu1.publish(MarkerOfPoints(tgt, true));
  pu2.publish(MarkerOfPoints(src, false));
  pu3.publish(MarkerOfPoints(TransformPoints(src, res1), false));
  pu4.publish(MarkerOfPoints(TransformPoints(src, res2), false));
  pub1.publish(MarkerOfNDT(m1.tmap(), {kRed, kCell, kCov}));

  int i;
  cout << "Index: ";
  while (cin >> i) {
    if (i >= 0 && i < (int)m1.tfs().size()) {
      pub2.publish(MarkerOfNDT(m1.smap(), {kGreen, kCell, kCov}, m1.tfs()[i]));
    }
    if (i >= 0 && i < (int)m2.tfs().size()) {
      pub3.publish(MarkerOfNDT(m2.smap(), {kGreen, kCell, kCov}, m2.tfs()[i]));
    }
    cout << "Index: ";
  }
}
