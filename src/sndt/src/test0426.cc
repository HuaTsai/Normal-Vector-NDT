#include <bits/stdc++.h>
#include <pcl_ros/point_cloud.h>
#include <nav_msgs/Path.h>
#include <rosbag/bag.h>
#include <boost/program_options.hpp>
#include <tf2_msgs/TFMessage.h>
#include "common/common.h"
#include "sndt/EgoPointClouds.h"
#include "sndt/wrapper.hpp"
#include "dbg/dbg.h"

using namespace std;
namespace po = boost::program_options;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
vector<common::MatchInternal> mits;

// PC0  PC1  PC2  ... PCn-2    PCn-1
//  t0   t1   t2  ...  tn-2     tn-1
//    v0   v1     ...      vn-2
PointCloud AugmentPointCloud(const vector<PointCloud> &pcs,
                             const vector<array<double, 3>> &vxyth,
                             const vector<ros::Time> &times,
                             Eigen::Matrix4f &Tn0) {
  PointCloud ret = pcs.back();
  int n = pcs.size();
  Tn0 = Eigen::Matrix4f::Identity();
  double dx = 0, dy = 0, dth = 0;
  for (int i = n - 2; i >= 0; --i) {
    double dt = (times[i + 1] - times[i]).toSec();
    dx += -vxyth[i][0] * dt;
    dy += -vxyth[i][1] * dt;
    dth += -vxyth[i][2] * dt;
    Tn0 = common::Matrix4fFromXYTRadian({dx, dy, dth});
    PointCloud tmp;
    pcl::transformPointCloud(pcs[i], tmp, Tn0);
    ret += tmp;
  }
  return ret;
}

geometry_msgs::PoseStamped MakePST(const ros::Time &time,
                                   const Eigen::Matrix4f &mtx) {
  geometry_msgs::PoseStamped ret;
  ret.header.frame_id = "map";
  ret.header.stamp = time;
  ret.pose = tf2::toMsg(common::Affine3dFromMatrix4f(mtx));
  return ret;
}

int main(int argc, char **argv) {
  string infile, outfolder;
  bool bagout;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("infile", po::value<string>(&infile)->required(), "EgoPointClouds path")
      ("outfolder", po::value<string>(&outfolder)->default_value("."), "Output folder path")
      ("bagout", po::value<bool>(&bagout)->default_value(true), "Write rosbag");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  rosbag::Bag bag;
  if (bagout) {
    bag.open("replay.bag", rosbag::bagmode::Write);
  }

  vector<sndt::EgoPointClouds> vepcs;
  common::SerializationInput(infile, vepcs);
  int n = vepcs.size(), frames = 7;
  n = n / frames * frames;

  vector<PointCloud> pcs;
  PointCloud prepc, curpc;
  Eigen::Matrix4f curTr = Eigen::Matrix4f::Identity();
  vector<geometry_msgs::PoseStamped> vp;

  for (int i = 0; i < n; i += frames) {
    vector<PointCloud> pcs;
    vector<array<double, 3>> vxyth;
    vector<ros::Time> times;
    ros::Time pretime, curtime;

    // i-f, ..., i-1 | i, i+1, ..., i+f-1 | i+f -> actual id
    // ..., ...,  m  | i, ..., ...,   n   | ... -> symbol id
    // -- frames  -- |
    for (int j = 0; j < frames; ++j) {
      auto vepc = vepcs[i + j];
      PointCloud pc;
      pcl::fromROSMsg(vepc.augpc, pc);
      pcs.push_back(pc);
      vxyth.push_back({vepc.ego_vx, vepc.ego_vy, vepc.ego_vth});
      times.push_back(vepc.stamp);
    }
    curtime = times.back();
    vxyth.pop_back();
    Eigen::Matrix4f Tni;
    curpc = AugmentPointCloud(pcs, vxyth, times, Tni);

    if (i == 0) { prepc = curpc; vp.push_back(MakePST(curtime, curTr)); continue; }

    /****** Get Tmn ******/
    int m = i - 1;
    pretime = vepcs[m].stamp;
    double dt = (vepcs[i].stamp - pretime).toSec();
    vector<double> xyt = {dt * vepcs[m].ego_vx, dt * vepcs[m].ego_vy, vepcs[m].ego_vth};
    auto Tmi = common::Matrix4fFromXYTRadian(xyt);
    auto Tmn = Tmi * Tni.inverse();


    /****** Match ******/
    static int matchid = 0;
    common::MatchPackage mp;
    mp.source = MatrixXdFromPCL(curpc);
    mp.target = MatrixXdFromPCL(prepc);
    if (matchid == 62) {
      dbg("idx62");
      Eigen::MatrixXd res;
      for (int ii = 0; ii < mp.target.cols(); ++ii) {
        double x = mp.target(0, ii), y = mp.target(1, ii);
        if (x > -120 && x < -100 && y > -10 && y < 40) {
          dbg("remove 1", x, y);
          continue;
        }
        res.conservativeResize(3, res.cols() + 1);
        res.col(res.cols() - 1) = mp.target.col(ii);
      }
      mp.target = res;
    }
    mp.guess = common::Matrix3dFromMatrix4f(Tmn);
    common::MatchInternal mit;
    DoSNDT(mp, mit, {3, 1, 2, 0.001});
    mits.push_back(mit);
    auto err = common::TransNormRotDegAbsFromMatrix3d(mp.result.inverse() * mp.guess);
    dbg(matchid, i, err);
    ++matchid;

    /****** Update ******/
    Eigen::Matrix4f preTr = curTr;
    curTr = curTr * common::Matrix4fFromMatrix3d(mp.result);
    vp.push_back(MakePST(curtime, curTr));

    /****** Write to bag ******/
    if (bagout) {
      // 1. prepc 2. curpc 3. base_link_pre tf 4. base_link tf 5. path
      sensor_msgs::PointCloud2 prepcmsg;
      pcl::toROSMsg(prepc, prepcmsg);
      prepcmsg.header.frame_id = "base_link_pre";
      prepcmsg.header.stamp = curtime;
      bag.write("prepc", curtime, prepcmsg);

      sensor_msgs::PointCloud2 curpcmsg;
      pcl::toROSMsg(curpc, curpcmsg);
      curpcmsg.header.frame_id = "base_link";
      curpcmsg.header.stamp = curtime;
      bag.write("curpc", curtime, curpcmsg);

      tf2_msgs::TFMessage tfmsg;
      auto pretst = common::TstFromMatrix4f(preTr, curtime, "map", "base_link_pre");
      auto curtst = common::TstFromMatrix4f(curTr, curtime, "map", "base_link");
      tfmsg.transforms.push_back(pretst);
      tfmsg.transforms.push_back(curtst);
      bag.write("tf", curtime, tfmsg);

      nav_msgs::Path path;
      path.header.frame_id = "map";
      path.header.stamp = curtime;
      path.poses = vp;
      bag.write("path", curtime, path);

      auto guess = tf2::eigenToTransform(common::Affine3dFromMatrix3d(mp.guess));
      auto result = tf2::eigenToTransform(common::Affine3dFromMatrix3d(mp.result));
      bag.write("guess", curtime, guess.transform);
      bag.write("result", curtime, result.transform);
    }

    prepc = curpc;
  }

  /****** Save MatchInternals ******/
  common::SerializeOut("mits.ser", mits);

  nav_msgs::Path path;
  path.header.frame_id = "map";
  path.header.stamp = vp[0].header.stamp;
  path.poses = vp;
  common::SerializationOutput(outfolder + "/path.ser", path);
  if (bagout) {
    bag.close();
  }
}
