#include <bits/stdc++.h>
#include <nav_msgs/Path.h>
#include "common/common.h"
#include "common/EgoPointClouds.h"
#include "pclmatch/wrapper.hpp"
#include "dbg/dbg.h"

using namespace std;
using namespace Eigen;
typedef pcl::PointCloud<pcl::PointXYZ> PCXYZ;

PCXYZ PointCloudFromPoints(const vector<geometry_msgs::Point> &pts) {
  PCXYZ ret;
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
PCXYZ AugmentPointCloud(const vector<PCXYZ> &pcs,
                             const vector<vector<double>> &vxyts,
                             const vector<ros::Time> &times,
                             Matrix4f &Tn0) {
  PCXYZ ret = pcs.back();
  int n = pcs.size();
  Tn0 = Matrix4f::Identity();
  double dx = 0, dy = 0, dth = 0;
  for (int i = n - 2; i >= 0; --i) {
    double dt = (times[i + 1] - times[i]).toSec();
    dx += -vxyts[i][0] * dt;
    dy += -vxyts[i][1] * dt;
    dth += -vxyts[i][2] * dt;
    dbg(dx, dy, dth);
    Tn0 = common::Matrix4fFromXYTRadian({dx, dy, dth});
    PCXYZ tmp;
    pcl::transformPointCloud(pcs[i], tmp, Tn0);
    ret += tmp;
  }
  return ret;
}

geometry_msgs::PoseStamped MakePST(const ros::Time &time,
                                   const Matrix4f &mtx) {
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
  vector<PCXYZ> pcs;
  PCXYZ prepc, curpc;
  Matrix4f Tr = Matrix4f::Identity();
  vector<geometry_msgs::PoseStamped> vp;
  for (int i = 0; i < n; i += frames) {
    vector<PCXYZ> pcs;
    vector<vector<double>> vxyts;
    vector<ros::Time> times;
    // i-f, ..., i-1 | i, i+1, ..., i+f-1 | i+f -> actual id
    // ..., ...,  m  | i, ..., ...,   n   | ... -> symbol id
    // -- frames  -- |
    for (int j = 0; j < frames; ++j) {
      auto vepc = vepcs[i + j];
      PCXYZ pc = PointCloudFromPoints(vepc.augpc);
      pcs.push_back(pc);
      vxyts.push_back(vepc.vxyt);
      times.push_back(vepc.stamp);
    }
    vxyts.pop_back();
    Matrix4f Tni;
    curpc = AugmentPointCloud(pcs, vxyts, times, Tni);

    if (i == 0) { prepc = curpc; vp.push_back(MakePST(times.front(), Tr)); continue; }

    /****** Get Tmn ******/
    int m = i - 1;
    double dt = (vepcs[i].stamp - vepcs[m].stamp).toSec();
    Vector3d xyt = dt * Vector3d::Map(vepcs[m].vxyt.data(), 3);
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
