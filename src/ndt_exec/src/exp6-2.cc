// 3D ICP
#include <common/common.h>
#include <metric/metric.h>
#include <nav_msgs/Path.h>
#include <ndt/matcher.h>
#include <ndt_exec/wrapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>

#include <boost/program_options.hpp>

using namespace std;

using namespace Eigen;
namespace po = boost::program_options;

struct Res {
  Res() : n(0), sc(0) {}
  void Show() {
    printf(" its: %.2f / %.2f\n", Stat(its).mean, Stat(its).max);
    printf(" err: %.4f / %.4f\n", Stat(terr).rms, Stat(rerr).rms);
    printf("  sr: %.2f (%d / %d)\n", sc / double(n) * 100., sc, n);
    printf("time: %.2f / %.2f / %.2f / %.2f / %.2f\n", Stat(nm).mean,
           Stat(ndt).mean, Stat(bud).mean, Stat(opt).mean, Stat(ttl).mean);
  }
  int n;
  vector<double> its;
  vector<double> terr;
  vector<double> rerr;
  vector<double> nm, ndt, bud, opt, ttl;
  int sc;
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

void PNumpy(vector<double> coll, string str) {
  cout << str << " = [";
  copy(coll.begin(), coll.end(), ostream_iterator<double>(cout, ", "));
  cout << "]\n";
}

void RunMatch(const vector<Vector3d> &src,
              const vector<Vector3d> &tgt,
              const Affine3d &ben,
              Matcher &m,
              Res &r) {
  m.SetSource(src);
  m.SetTarget(tgt);
  auto res = m.Align();
  auto err = TransNormRotDegAbsFromAffine3d(res * ben.inverse());
  ++r.n;
  r.its.push_back(m.iteration());
  r.terr.push_back(err(0));
  r.rerr.push_back(err(1));
  r.nm.push_back(m.timer().normal() / 1e3);
  r.ndt.push_back(m.timer().ndt() / 1e3);
  r.bud.push_back(m.timer().build() / 1e3);
  r.opt.push_back(m.timer().optimize() / 1e3);
  r.ttl.push_back(m.timer().total() / 1e3);
  if (err(0) < 0.2 && err(1) < 3) ++r.sc;
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
  SerializationInput(JoinPath(GetDataPath(d), "lidar.ser"), vpc);

  auto t0 = vpc[0].header.stamp;
  GtLocal(gt, t0);
  Res r1, r2;
  std::vector<double> bent, benr;
  if (n == -1) n = vpc.size() - 1;
  tqdm bar;
  for (int i = 0; i < n; i += f) {
    bar.progress(i, n);
    auto tgt = PCMsgTo3D(vpc[i], 0);
    auto src = PCMsgTo3D(vpc[i + f], 0);
    TransformPointsInPlace(tgt, aff3);
    TransformPointsInPlace(src, aff3);
    auto ben = BM(gt, vpc[i].header.stamp, vpc[i + f].header.stamp);
    bent.push_back(TransNormRotDegAbsFromAffine3d(ben)(0));
    benr.push_back(TransNormRotDegAbsFromAffine3d(ben)(1));

    auto m1 = ICPMatcher::GetBasic({});
    RunMatch(src, tgt, ben, m1, r1);

    auto m2 = SICPMatcher::GetBasic({}, 0.5);
    RunMatch(src, tgt, ben, m2, r2);
  }
  bar.finish();

  r1.Show();
  r2.Show();
}

// nav_msgs::Path InitFirstPose(const ros::Time &time) {
//   nav_msgs::Path path;
//   path.header.frame_id = "map";
//   path.header.stamp = time;
//   path.poses.push_back(MakePoseStampedMsg(time,
//   Eigen::Affine3d::Identity())); return path;
// }

// struct Res {
//   Res() : Tr(Affine3d::Identity()) {}
//   void Show() {
//     printf(" its: %f / %f\n", Stat(its).mean, Stat(its).max);
//     printf(" err: %f / %f\n", Stat(terr).rms, Stat(rerr).rms);
//     printf("corr: %f\n", Stat(corr).mean);
//     double den = its.size() * 1000.;
//     printf("time: %f / %f\n", timer.optimize() / den, timer.total() / den);
//   }
//   vector<double> its;
//   vector<double> terr;
//   vector<double> rerr;
//   vector<double> corr;
//   vector<double> opt;
//   Timer timer;
//   nav_msgs::Path path;
//   Affine3d Tr;
// };

// Affine3d BM(const nav_msgs::Path &gt,
//             const ros::Time &t1,
//             const ros::Time &t2) {
//   Affine3d To, Ti;
//   tf2::fromMsg(GetPose(gt.poses, t2), To);
//   tf2::fromMsg(GetPose(gt.poses, t1), Ti);
//   return Ti.inverse() * To;
// }

// void GtLocal(nav_msgs::Path &path, const ros::Time &start) {
//   auto startpose = GetPose(path.poses, start);
//   Eigen::Affine3d preT;
//   tf2::fromMsg(startpose, preT);
//   preT = preT.inverse();
//   for (size_t i = 0; i < path.poses.size(); ++i) {
//     Eigen::Affine3d T;
//     tf2::fromMsg(path.poses[i].pose, T);
//     Eigen::Affine3d newT = preT * T;
//     path.poses[i].pose = tf2::toMsg(newT);
//   }
// }

// void PrintRes(const nav_msgs::Path &est, const nav_msgs::Path &gt) {
//   TrajectoryEvaluation te;
//   te.set_estpath(est);
//   te.set_gtpath(gt);
//   te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeByLength);
//   te.set_length(100);
//   auto rpe = te.ComputeRMSError2D();
//   cout << rpe.first.rms << " / " << rpe.second.rms << endl;
// }

// void PNumpy(vector<double> coll, string str) {
//   cout << str << " = [";
//   copy(coll.begin(), coll.end(), ostream_iterator<double>(cout, ", "));
//   cout << "]\n";
// }

// int main(int argc, char **argv) {
//   Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
//                   Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
//   string d;
//   int f, n;
//   po::options_description desc("Allowed options");
//   // clang-format off
//   desc.add_options()
//       ("h,h", "Produce help message")
//       ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
//       ("f,f", po::value<int>(&f)->required()->default_value(1), "Frames")
//       ("n,n", po::value<int>(&n)->default_value(-1), "Frames");
//   // clang-format on
//   po::variables_map vm;
//   po::store(po::parse_command_line(argc, argv, desc), vm);
//   if (vm.count("help")) {
//     cout << desc << endl;
//     return 1;
//   }
//   po::notify(vm);

//   nav_msgs::Path gt;
//   SerializationInput(JoinPath(GetDataPath(d), "gt.ser"), gt);
//   vector<sensor_msgs::PointCloud2> vpc;
//   SerializationInput(JoinPath(GetDataPath(d), "lidar.ser"), vpc);

//   auto t0 = vpc[0].header.stamp;
//   GtLocal(gt, t0);
//   Res r1, r2;
//   r1.path = InitFirstPose(t0);
//   r2.path = InitFirstPose(t0);
//   if (n == -1) n = vpc.size() - 1;
//   tqdm bar;
//   for (int i = 0; i < n; i += f) {
//     bar.progress(i, n);
//     auto tgt = PCMsgTo3D(vpc[i], 0);
//     auto src = PCMsgTo3D(vpc[i + f], 0);
//     auto tj = vpc[i + f].header.stamp;
//     TransformPointsInPlace(tgt, aff3);
//     TransformPointsInPlace(src, aff3);
//     auto ben = BM(gt, vpc[i].header.stamp, vpc[i + f].header.stamp);

//     auto m1 = ICPMatcher::GetBasic({});
//     m1.SetSource(src);
//     m1.SetTarget(tgt);
//     auto res1 = m1.Align();
//     auto err1 = TransNormRotDegAbsFromAffine3d(res1 * ben.inverse());
//     r1.terr.push_back(err1(0));
//     r1.rerr.push_back(err1(1));
//     r1.corr.push_back(m1.corres());
//     r1.timer += m1.timer();
//     r1.opt.push_back(m1.timer().optimize() / 1000.);
//     r1.its.push_back(m1.iteration());
//     r1.Tr = r1.Tr * res1;
//     r1.path.poses.push_back(MakePoseStampedMsg(tj, r1.Tr));

//     auto m2 = SICPMatcher::GetBasic({}, 0.5);
//     m2.SetSource(src);
//     m2.SetTarget(tgt);
//     auto res2 = m2.Align();
//     auto err2 = TransNormRotDegAbsFromAffine3d(res2 * ben.inverse());
//     r2.terr.push_back(err2(0));
//     r2.rerr.push_back(err2(1));
//     r2.corr.push_back(m2.corres());
//     r2.timer += m2.timer();
//     r2.opt.push_back(m2.timer().optimize() / 1000.);
//     r2.its.push_back(m2.iteration());
//     r2.Tr = r2.Tr * res2;
//     r2.path.poses.push_back(MakePoseStampedMsg(tj, r2.Tr));
//   }
//   bar.finish();

//   ros::init(argc, argv, "exp6_2");
//   ros::NodeHandle nh;
//   auto pub1 = nh.advertise<nav_msgs::Path>("path1", 0, true);
//   auto pub3 = nh.advertise<nav_msgs::Path>("path3", 0, true);
//   auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);
//   pub1.publish(r1.path);
//   pub3.publish(r2.path);
//   pubgt.publish(gt);

//   cout << "-------------" << endl;
//   r1.Show();
//   cout << "-------------" << endl;
//   r2.Show();
//   cout << endl;
//   r1.timer.Show();
//   r2.timer.Show();

//   printf("%.2f, %.2f\n%.2f, %.2f\n",
//          (r1.timer.build() + r1.timer.optimize()) / 1000. / n,
//          r1.timer.total() / 1000. / n,
//          (r2.timer.build() + r2.timer.optimize()) / 1000. / n,
//          r2.timer.total() / 1000. / n);

//   ros::spin();
// }
