// PCL Results
#include <common/common.h>
#include <metric/metric.h>
#include <nav_msgs/Path.h>
#include <ndt/matcher.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl_ros/point_cloud.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>
#include <boost/program_options.hpp>

using namespace std;

using namespace Eigen;
namespace po = boost::program_options;

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
  vector<double> opt;
  Timer timer;
  nav_msgs::Path path;
  Affine3d Tr;
};

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
  int f, n;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
      ("f,f", po::value<int>(&f)->required()->default_value(1), "Frames")
      ("n,n", po::value<int>(&n)->default_value(-1), "Frames");
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
  SerializationInput(JoinPath(GetDataPath(d), "lidarall.ser"), vpc);

  map<double, int, greater<>> mp1, mp2, mp3, mp4;
  auto t0 = vpc[0].header.stamp;
  GtLocal(gt, t0);
  Res r1, r2;
  r1.path = InitFirstPose(t0);
  r2.path = InitFirstPose(t0);
  std::vector<double> bent, benr;
  if (n == -1) n = vpc.size() - 1;
  tqdm bar;
  for (int i = 0; i < n; i += f) {
    auto tj = vpc[i + f].header.stamp;
    auto ben = BM(gt, vpc[i].header.stamp, vpc[i + f].header.stamp);
    bent.push_back(TransNormRotDegAbsFromAffine3d(ben)(0));
    benr.push_back(TransNormRotDegAbsFromAffine3d(ben)(1));

    double voxel = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto pt : TransformPoints(PCMsgTo3D(vpc[i + f], voxel), aff3)) {
      pcl::PointXYZ p;
      p.x = pt(0), p.y = pt(1), p.z = pt(2);
      src->push_back(p);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto pt : TransformPoints(PCMsgTo3D(vpc[i], voxel), aff3)) {
      pcl::PointXYZ p;
      p.x = pt(0), p.y = pt(1), p.z = pt(2);
      tgt->push_back(p);
    }
    pcl::PointCloud<pcl::PointXYZ> out;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    auto t1 = GetTime();
    // icp.align(out, ben.matrix().cast<float>());
    icp.align(out);
    auto t2 = GetTime();
    auto res1 = Affine3d(icp.getFinalTransformation().cast<double>());
    auto err1 = TransNormRotDegAbsFromAffine3d(res1 * ben.inverse());
    r1.terr.push_back(err1(0));
    r1.rerr.push_back(err1(1));
    r1.opt.push_back(GetDiffTime(t1, t2) / 1000.);
    r1.Tr = r1.Tr * res1;
    r1.path.poses.push_back(MakePoseStampedMsg(tj, r1.Tr));

    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setInputSource(src);
    ndt.setInputTarget(tgt);
    ndt.setResolution(1);
    auto t3 = GetTime();
    // ndt.align(out, ben.matrix().cast<float>());
    ndt.align(out);
    auto t4 = GetTime();
    auto res2 = Affine3d(ndt.getFinalTransformation().cast<double>());
    auto err2 = TransNormRotDegAbsFromAffine3d(res2 * ben.inverse());
    r2.terr.push_back(err2(0));
    r2.rerr.push_back(err2(1));
    r2.opt.push_back(GetDiffTime(t3, t4) / 1000.);
    r2.Tr = r2.Tr * res2;
    r2.path.poses.push_back(MakePoseStampedMsg(tj, r2.Tr));
  }
  bar.finish();
  cout << *max_element(r1.terr.begin(), r1.terr.end()) << " -> ";
  Print(LargestNIndices(r1.terr, 3), "t1");
  cout << *max_element(r1.rerr.begin(), r1.rerr.end()) << " -> ";
  Print(LargestNIndices(r1.rerr, 3), "r1");
  cout << *max_element(r2.terr.begin(), r2.terr.end()) << " -> ";
  Print(LargestNIndices(r2.terr, 3), "t2");
  cout << *max_element(r2.rerr.begin(), r2.rerr.end()) << " -> ";
  Print(LargestNIndices(r2.rerr, 3), "r2");

  vector<double> w1(r1.opt.size());
  transform(r2.opt.begin(), r2.opt.end(), r1.opt.begin(), w1.begin(),
            std::minus<double>());
  cout << *max_element(w1.begin(), w1.end()) << " -> ";
  Print(LargestNIndices(w1, 5), "NDT Win");

  vector<double> w2(r1.opt.size());
  transform(r1.opt.begin(), r1.opt.end(), r2.opt.begin(), w2.begin(),
            std::minus<double>());
  cout << *max_element(w2.begin(), w2.end()) << " -> ";
  Print(LargestNIndices(w2, 5), "NNDT Win");

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
  // PNumpy(r1.opt, "opt1");
  // PNumpy(r2.terr, "t2");
  // PNumpy(r2.rerr, "r2");
  // PNumpy(r2.opt, "opt2");
  PrintRes(r1.path, gt);
  PrintRes(r2.path, gt);
  cout << "-------------" << endl;
  r1.Show();
  cout << "time average: " << Stat(r1.opt).mean << endl;
  cout << "-------------" << endl;
  r2.Show();
  cout << "time average: " << Stat(r2.opt).mean << endl;
  cout << endl;
  ros::spin();
}
