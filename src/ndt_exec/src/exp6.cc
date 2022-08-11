// 3D NDT
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
              NDTMatcher &m,
              Res &r) {
  m.set_intrinsic(0.00005);
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

void RunMatch2(const vector<Vector3d> &src,
               const vector<Vector3d> &tgt,
               const Affine3d &ben,
               NDTMatcher &m,
               Res &r) {
  m.set_intrinsic(0.00005);
  m.SetSource(src);
  m.SetTarget(tgt);
  auto res = m.Align();

  auto m2 = ICPMatcher::GetBasic({kOneTime});
  m2.SetSource(src);
  m2.SetTarget(tgt);
  auto res2 = m2.Align(res);
  auto err = TransNormRotDegAbsFromAffine3d(res2 * ben.inverse());

  ++r.n;
  r.its.push_back((m.iteration() + m2.iteration()));
  r.terr.push_back(err(0));
  r.rerr.push_back(err(1));
  r.nm.push_back((m.timer().normal() + m2.timer().normal()) / 1e3);
  r.ndt.push_back((m.timer().ndt() + m2.timer().ndt()) / 1e3);
  r.bud.push_back((m.timer().build() + m2.timer().build()) / 1e3);
  r.opt.push_back((m.timer().optimize() + m2.timer().optimize()) / 1e3);
  r.ttl.push_back((m.timer().total() + m2.timer().total()) / 1e3);
  if (err(0) < 0.2 && err(1) < 3) ++r.sc;
}

void RunMatch3(const vector<Vector3d> &src,
               const vector<Vector3d> &tgt,
               const Affine3d &ben,
               NDTMatcher &m,
               Res &r) {
  m.set_intrinsic(0.00005);
  m.SetSource(src);
  m.SetTarget(tgt);
  auto res = m.Align();

  unordered_set<Options> op = {kNDT,      k1to1,     kPointCov,
                               kAnalytic, kNoReject, kOneTime};
  auto m2 = NDTMatcher::GetBasic(op, 1, 0.05);
  m2.set_intrinsic(0.00005);
  m2.SetSource(src);
  m2.SetTarget(tgt);
  auto res2 = m2.Align(res);
  auto err = TransNormRotDegAbsFromAffine3d(res2 * ben.inverse());

  ++r.n;
  r.its.push_back((m.iteration() + m2.iteration()));
  r.terr.push_back(err(0));
  r.rerr.push_back(err(1));
  r.nm.push_back((m.timer().normal()) / 1e3);
  r.ndt.push_back((m.timer().ndt()) / 1e3);
  r.bud.push_back((m.timer().build() + m2.timer().build()) / 1e3);
  r.opt.push_back((m.timer().optimize() + m2.timer().optimize()) / 1e3);
  r.ttl.push_back((m.timer().total() + m2.timer().total()) / 1e3);
  if (err(0) < 0.2 && err(1) < 3) ++r.sc;
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  string d;
  int f, n;
  double ndtd2, nndtd2, cs;
  bool orj;
  bool two;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
      ("f,f", po::value<int>(&f)->required()->default_value(1), "Frames")
      ("n,n", po::value<int>(&n)->default_value(-1), "Frames")
      ("c,c", po::value<double>(&cs)->default_value(-1), "Cell size")
      ("o,o", po::value<bool>(&orj)->default_value(false)->implicit_value(true), "Reject")
      ("t,t", po::value<bool>(&two)->default_value(false)->implicit_value(true), "Two Stages")
      ("ndtd2,a", po::value<double>(&ndtd2)->default_value(0.05), "d2 NDT")
      ("nndtd2,b", po::value<double>(&nndtd2)->default_value(0.05), "d2 NNDT");
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

    unordered_set<Options> op1 = {kNDT, k1to1, kPointCov, kAnalytic};
    if (!orj) op1.insert(kNoReject);
    if (two) {
      // auto m1 = NDTMatcher::GetIter(op1, {0.5, 1, 2, 3}, ndtd2);
      // RunMatch(src, tgt, ben, m1, r1);
    } else {
      auto m1 = NDTMatcher::GetBasic(op1, cs, ndtd2);
      RunMatch(src, tgt, ben, m1, r1);
    }

    unordered_set<Options> op2 = {kNVNDT, k1to1, kPointCov, kAnalytic};
    if (!orj) op2.insert(kNoReject);
    if (two) {
      // auto m2 = NDTMatcher::GetBasic(op2, cs, nndtd2);
      // RunMatch2(src, tgt, ben, m2, r2);
      auto m2 = NDTMatcher::GetBasic(op2, cs, nndtd2);
      RunMatch3(src, tgt, ben, m2, r2);
    } else {
      auto m2 = NDTMatcher::GetBasic(op2, cs, nndtd2);
      RunMatch(src, tgt, ben, m2, r2);
    }
  }
  bar.finish();

  r1.Show();
  r2.Show();
}
