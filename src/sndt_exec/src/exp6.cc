// 3D NDT
#include <common/common.h>
#include <metric/metric.h>
#include <nav_msgs/Path.h>
#include <ndt/matcher.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>

using namespace std;

using namespace Eigen;
namespace po = boost::program_options;
NDTMatcher::MatchType kNDTLS = NDTMatcher::MatchType::kNDTLS;
NDTMatcher::MatchType kNDTTR = NDTMatcher::MatchType::kNDTTR;
NDTMatcher::MatchType kNNDTLS = NDTMatcher::MatchType::kNNDTLS;
NDTMatcher::MatchType kNNDTTR = NDTMatcher::MatchType::kNNDTTR;

nav_msgs::Path InitFirstPose(const ros::Time &time) {
  nav_msgs::Path path;
  path.header.frame_id = "map";
  path.header.stamp = time;
  path.poses.push_back(MakePoseStampedMsg(time, Eigen::Affine3d::Identity()));
  return path;
}

struct Res {
  Res() : Tr(Affine3d::Identity()) {}
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
  nav_msgs::Path path;
  Affine3d Tr;
};

std::vector<Eigen::Vector3d> PCMsgTo3D(const sensor_msgs::PointCloud2 &msg,
                                       double voxel) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(msg, *pc);
  if (voxel != 0) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pc);
    vg.setLeafSize(voxel, voxel, voxel);
    vg.filter(*pc);
  }
  std::vector<Eigen::Vector3d> ret;
  for (const auto &pt : *pc)
    if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
      ret.push_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
  return ret;
}

Affine3d BM(const nav_msgs::Path &gt,
            const ros::Time &t1,
            const ros::Time &t2) {
  Affine3d To, Ti;
  tf2::fromMsg(GetPose(gt.poses, t2), To);
  tf2::fromMsg(GetPose(gt.poses, t1), Ti);
  return Ti.inverse() * To;
}

void GtLocal(nav_msgs::Path &path, const ros::Time &start) {
  auto startpose = GetPose(path.poses, start);
  Eigen::Affine3d preT;
  tf2::fromMsg(startpose, preT);
  preT = preT.inverse();
  for (size_t i = 0; i < path.poses.size(); ++i) {
    Eigen::Affine3d T;
    tf2::fromMsg(path.poses[i].pose, T);
    Eigen::Affine3d newT = preT * T;
    path.poses[i].pose = tf2::toMsg(newT);
  }
}

void PrintRes(const nav_msgs::Path &est, const nav_msgs::Path &gt) {
  TrajectoryEvaluation te;
  te.set_estpath(est);
  te.set_gtpath(gt);
  te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeByLength);
  te.set_length(100);
  auto rpe = te.ComputeRMSError2D();
  cout << rpe.first.rms << " / " << rpe.second.rms << endl;
}

void PNumpy(vector<double> coll, string str) {
  cout << str << " = [";
  copy(coll.begin(), coll.end(), ostream_iterator<double>(cout, ", "));
  cout << "]\n";
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  string d;
  bool tr;
  int f, n;
  double cs, ndtd2, nndtd2;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
      ("f,f", po::value<int>(&f)->required()->default_value(1), "Frames")
      ("n,n", po::value<int>(&n)->default_value(-1), "Frames")
      ("ndtd2,a", po::value<double>(&ndtd2)->default_value(0.3), "d2 NDT")
      ("nndtd2,b", po::value<double>(&nndtd2)->default_value(0.3), "d2 NNDT")
      ("cs,c", po::value<double>(&cs)->default_value(1), "Cell Size")
      ("tr,t", po::value<bool>(&tr)->default_value(false)->implicit_value(true), "Trust Region");
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

  auto t0 = vpc[0].header.stamp;
  GtLocal(gt, t0);
  Res r1, r2;
  r1.path = InitFirstPose(t0);
  r2.path = InitFirstPose(t0);
  std::vector<double> bent, benr;
  if (n == -1) n = vpc.size() - 1;
  // for (size_t i = 0; i < vpc.size() - 1; ++i) {
  tqdm bar;
  for (int i = 0; i < n; i += f) {
    // bar.progress(i, n);
    auto tgt = PCMsgTo3D(vpc[i], 0);
    auto src = PCMsgTo3D(vpc[i + f], 0);
    auto tj = vpc[i + f].header.stamp;
    TransformPointsInPlace(tgt, aff3);
    TransformPointsInPlace(src, aff3);
    auto ben = BM(gt, vpc[i].header.stamp, vpc[i + f].header.stamp);
    bent.push_back(TransNormRotDegAbsFromAffine3d(ben)(0));
    benr.push_back(TransNormRotDegAbsFromAffine3d(ben)(1));

    // cout << "NDT" << endl;
    NDTMatcher m1(tr ? kNDTTR : kNDTLS, cs, ndtd2);
    m1.SetSource(src);
    m1.SetTarget(tgt);
    auto res1 = m1.Align();
    // ADD
    // auto res11 = m1.Align();
    // NDTMatcher m12(tr ? kNDTTR : kNDTLS, 1, ndtd2);
    // m12.SetSource(src);
    // m12.SetTarget(tgt);
    // auto res1 = m12.Align(res11);
    // ENDANDD
    auto err1 = TransNormRotDegAbsFromAffine3d(res1 * ben.inverse());

    r1.terr.push_back(err1(0));
    r1.rerr.push_back(err1(1));
    r1.corr.push_back(m1.corres());
    r1.timer += m1.timer();
    r1.its.push_back(m1.iteration());
    r1.Tr = r1.Tr * res1;
    r1.path.poses.push_back(MakePoseStampedMsg(tj, r1.Tr));

    // cout << "NNDT" << endl;
    NDTMatcher m2(tr ? kNNDTTR : kNNDTLS, cs, nndtd2);
    m2.SetSource(src);
    m2.SetTarget(tgt);
    auto res2 = m2.Align();

    // ADD
    // auto res21 = m2.Align();
    // NDTMatcher m22(tr ? kNDTTR : kNDTLS, 1, ndtd2);
    // m22.SetSource(src);
    // m22.SetTarget(tgt);
    // auto res2 = m22.Align(res21);
    // ENDANDD

    auto err2 = TransNormRotDegAbsFromAffine3d(res2 * ben.inverse());
    r2.terr.push_back(err2(0));
    r2.rerr.push_back(err2(1));
    r2.corr.push_back(m2.corres());
    r2.timer += m2.timer();
    r2.its.push_back(m2.iteration());
    r2.Tr = r2.Tr * res2;
    r2.path.poses.push_back(MakePoseStampedMsg(tj, r2.Tr));
  }
  bar.finish();
  // cout << "tgt: ";
  // Stat(bent).PrintResult();
  // cout << "rgt: ";
  // Stat(benr).PrintResult();
  ros::init(argc, argv, "exp6");
  ros::NodeHandle nh;
  auto pub5 = nh.advertise<nav_msgs::Path>("path5", 0, true);
  auto pub7 = nh.advertise<nav_msgs::Path>("path7", 0, true);
  auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);
  pub5.publish(r1.path);
  pub7.publish(r2.path);
  pubgt.publish(gt);

  // PNumpy(r1.terr, "t1");
  // PNumpy(r1.rerr, "r1");
  // PNumpy(r2.terr, "t2");
  // PNumpy(r2.rerr, "r2");
  // PNumpy(r1.its, "it1");
  // PNumpy(r2.its, "it2");
  // cout << "err(t1, r1, t2, r2)" << endl;
  PrintRes(r1.path, gt);
  PrintRes(r2.path, gt);
  cout << "-------------" << endl;
  r1.Show();
  cout << "-------------" << endl;
  r2.Show();
  cout << endl;
  ros::spin();
}
