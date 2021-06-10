#include <bits/stdc++.h>
#include <nav_msgs/Path.h>
#include "common/common.h"
#include "common/EgoPointClouds.h"
#include "pclmatch/wrapper.hpp"
#include "dbg/dbg.h"

using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

PointCloud PointCloudFromPoints(const vector<geometry_msgs::Point> &pts) {
  PointCloud ret;
  for (auto &pt : pts) {
    pcl::PointXYZ p;
    p.x = pt.x;
    p.y = pt.y;
    p.z = pt.z;
    ret.push_back(p);
  }
  return ret;
}

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
    dbg(dx, dy, dth);
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
  vector<common::EgoPointClouds> vepcs;
  common::SerializationInput(argv[1], vepcs);
  int n = vepcs.size(), frames = 5;
  n = n / frames * frames;
  vector<PointCloud> pcs;
  PointCloud prepc, curpc;
  Eigen::Matrix4f Tr = Eigen::Matrix4f::Identity();
  vector<geometry_msgs::PoseStamped> vp;
  for (int i = 0; i < n; i += frames) {
    vector<PointCloud> pcs;
    vector<array<double, 3>> vxyth;
    vector<ros::Time> times;
    // i-f, ..., i-1 | i, i+1, ..., i+f-1 | i+f -> actual id
    // ..., ...,  m  | i, ..., ...,   n   | ... -> symbol id
    // -- frames  -- |
    for (int j = 0; j < frames; ++j) {
      auto vepc = vepcs[i + j];
      PointCloud pc = PointCloudFromPoints(vepc.augpc);
      pcs.push_back(pc);
      vxyth.push_back({vepc.ego_vx, vepc.ego_vy, vepc.ego_vth});
      times.push_back(vepc.stamp);
    }
    vxyth.pop_back();
    Eigen::Matrix4f Tni;
    curpc = AugmentPointCloud(pcs, vxyth, times, Tni);

    if (i == 0) { prepc = curpc; vp.push_back(MakePST(times.front(), Tr)); continue; }

    /****** Get Tmn ******/
    int m = i - 1;
    double dt = (vepcs[i].stamp - vepcs[m].stamp).toSec();
    vector<double> xyt = {dt * vepcs[m].ego_vx, dt * vepcs[m].ego_vy, vepcs[m].ego_vth};
    auto Tmi = common::Matrix4fFromXYTRadian(xyt);
    auto Tmn = Tmi * Tni.inverse();
    // dbg(i, xyt, Tmi, Tmn, Tni);

    /****** Match ******/
    common::MatchPackage mp;
    mp.source = MatrixXdFromPCL(curpc);
    mp.target = MatrixXdFromPCL(prepc);
    mp.guess = common::Matrix3dFromMatrix4f(Tmn);
    common::MatchInternal mit;
    // auto guess = common::XYTDegreeFromMatrix3d(mp.guess);
    DoSICP(mp, {0});
    // auto res = common::XYTDegreeFromMatrix3d(mp.result);
    auto err = common::TransNormRotDegAbsFromMatrix3d(mp.result.inverse() * mp.guess);
    dbg(i, err);
    // mp.result = mp.guess;

    /****** Update ******/
    Tr = Tr * common::Matrix4fFromMatrix3d(mp.result);
    vp.push_back(MakePST(times.front(), Tr));

    prepc = curpc;
  }
  nav_msgs::Path path;
  path.header.frame_id = "map";
  path.header.stamp = vp[0].header.stamp;
  path.poses = vp;
  common::SerializationOutput(APATH(20210422/pathsicp.ser), path);
}
