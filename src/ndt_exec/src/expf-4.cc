// nuScenes, NDT, different bags, overall

#include <common/common.h>
#include <metric/metric.h>
#include <nav_msgs/Path.h>
#include <ndt/matcher.h>
#include <ndt_exec/groundfit.h>
#include <ndt_exec/wrapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <rosbag/view.h>
#include <tf2_msgs/TFMessage.h>

#include <boost/program_options.hpp>

using namespace std;

using namespace Eigen;
namespace po = boost::program_options;

// struct Res {
//   Res() : sc(0) {}
//   void Show() {
//     printf(" its: %.2f / %.2f\n", Stat(its).mean, Stat(its).max);
//     printf(" err: %.4f / %.4f\n", Stat(terr).rms, Stat(rerr).rms);
//     printf("  sr: %.2f\n", sc / double(its.size()) * 100.);
//     double den = its.size() * 1000.;
//     printf("time: %.2f / %.2f / %.2f / %.2f / %.2f\n", timer.normal() / den,
//            timer.ndt() / den, timer.build() / den, timer.optimize() / den,
//            timer.total() / den);
//   }
//   vector<double> its;
//   vector<double> terr;
//   vector<double> rerr;
//   Timer timer;
//   int sc;
// };

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

Affine3d GT(const vector<geometry_msgs::PoseStamped> &psts,
            const ros::Time &t1,
            const ros::Time &t2) {
  Affine3d To, Ti;
  tf2::fromMsg(GetPose(psts, t2), To);
  tf2::fromMsg(GetPose(psts, t1), Ti);
  return Ti.inverse() * To;
}

void RunMatch(const vector<Vector3d> &src,
              const vector<Vector3d> &tgt,
              const Affine3d &gt,
              NDTMatcher &m,
              Res &r) {
  m.set_intrinsic(0.00005);
  m.SetSource(src);
  m.SetTarget(tgt);
  auto res = m.Align();
  auto err = TransNormRotDegAbsFromAffine3d(res * gt.inverse());
  // r.terr.push_back(err(0));
  // r.rerr.push_back(err(1));
  // if (err(0) < 0.2 && err(1) < 3) ++r.sc;
  // r.timer += m.timer();
  // r.its.push_back(m.iteration());
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

void PNumpy(ofstream &fout, vector<double> coll, string str) {
  fout << str << " = [";
  copy(coll.begin(), coll.end(), ostream_iterator<double>(fout, ", "));
  fout << "]\n";
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  double d2 = 0.05;
  double cs;
  bool orj;
  vector<int> logids;

  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<vector<int>>(&logids)->required()->multitoken(), "lognum1 lognum2 ...")
      ("c,c", po::value<double>(&cs)->required(), "Cell Size")
      ("o,o", po::value<bool>(&orj)->default_value(false)->implicit_value(true), "Reject");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  GroundPlaneFit gpf;
  int a = 0;
  Res r1, r2;
  for (const auto &logid : logids) {
    for (const auto &fn : GetBagsPath(logid)) {
      printf("bag(%d): %s ... ", ++a, fn.c_str());
      fflush(stdout);
      // Fetch Data
      rosbag::Bag bag;
      bag.open(fn);
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

        tf2_msgs::TFMessageConstPtr tfmsg =
            m.instantiate<tf2_msgs::TFMessage>();
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

      // Match
      int cnt = 0;
      for (size_t i = 0; i < vpc.size() - 1u; ++i) {
        auto tgt = vpc[i];
        auto src = vpc[i + 1];
        if (ts[i] < psts[0].header.stamp ||
            ts[i + 1] > psts.back().header.stamp)
          continue;
        auto gt = GT(psts, ts[i], ts[i + 1]);

        unordered_set<Options> op1 = {kNDT, k1to1, kPointCov, kAnalytic};
        if (!orj) op1.insert(kNoReject);
        auto m1 = NDTMatcher::GetBasic(op1, cs, d2);
        RunMatch(src, tgt, gt, m1, r1);

        unordered_set<Options> op2 = {kNVNDT, k1to1, kPointCov, kAnalytic};
        if (!orj) op2.insert(kNoReject);
        auto m2 = NDTMatcher::GetBasic(op2, cs, d2);
        RunMatch(src, tgt, gt, m2, r2);
        ++cnt;
      }

      printf("%d\n", cnt);
      fflush(stdout);
    }
  }
  r1.Show();
  cout << "/-------------/" << endl;
  r2.Show();

  ofstream fout("expf-4.txt");
  PNumpy(fout, r1.terr, "t1");
  PNumpy(fout, r1.rerr, "r1");
  PNumpy(fout, r2.terr, "t2");
  PNumpy(fout, r2.rerr, "r2");
}
