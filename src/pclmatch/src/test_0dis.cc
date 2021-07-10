/**
 * @file test_0dis.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Evaluate Distribution of SICP and SNDT
 * @version 0.1
 * @date 2021-07-07
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <bits/stdc++.h>
#include <common/EgoPointClouds.h>
#include <common/common.h>
#include <pclmatch/wrapper.hpp>

// #include <boost/program_options.hpp>

// namespace po = boost::program_options;
using namespace std;
using namespace Eigen;
using namespace visualization_msgs;

MatrixXd Augment(const vector<common::EgoPointClouds> &vepcs, int start, int end, Affine2d &T) {
  int n = 0;
  for (int i = start; i <= end; ++i)
    n += vepcs[i].augpc.size();
  MatrixXd ret(2, n);
  double dx = 0, dy = 0, dth = 0;
  for (int i = start, j = 0; i <= end; ++i) {
    Affine2d T0i = Rotation2Dd(dth) * Translation2d(dx, dy);
    for (const auto &pt : vepcs[i].augpc) {
      Vector2d pt2(pt.x, pt.y);
      if (pt2.allFinite())
        pt2 = T0i * pt2;
      ret.col(j++) = pt2;
    }
    double dt = (vepcs[i + 1].stamp - vepcs[i].stamp).toSec();
    dx += vepcs[i].vxyt[0] * dt;
    dy += vepcs[i].vxyt[1] * dt;
    dth += vepcs[i].vxyt[2] * dt;
  }
  T = Rotation2Dd(dth) * Translation2d(dx, dy);
  return ret;
}

int main(int argc, char **argv) {
  string path = "/home/ee904/Desktop/HuaTsai/NormalNDT/Analysis/1Data/vepcs62-1.ser";
  int start_frame = 450, frames = 5;
  double radius = 2.5;
  // po::options_description desc("Allowed options");
  // desc.add_options()
  //     ("help,h", "Produce help message")
  //     ("datapath,p", po::value<string>(&path)->required(), "Data Path")
  //     ("startframe,s", po::value<int>(&start_frame)->default_value(6), "Start Frame")
  //     ("frames,f", po::value<int>(&frames)->default_value(5), "Frames")
  //     ("radius,r", po::value<double>(&radius)->default_value(2.5), "Radius");
  // po::variables_map vm;
  // po::store(po::parse_command_line(argc, argv, desc), vm);
  // po::notify(vm);
  // if (vm.count("help")) {
  //   cout << desc << endl;
  //   return 1;
  // }

  vector<common::EgoPointClouds> vepcs;  
  common::SerializationInput(path, vepcs);
  cout << vepcs.size() << endl;
  vector<Affine2d> T611s, T1116s;
  Affine2d T611, T1116;
  auto datat = Augment(vepcs, start_frame, start_frame + frames - 1, T611);
  auto mesht = MakeMesh(datat, {radius});
  auto kdt = new trimesh::KDtree(mesht->vertices);
  auto datas = Augment(vepcs, start_frame + frames, start_frame + 2 * frames - 1, T1116);
  auto meshs = MakeMesh(datas, {radius});
  auto kds = new trimesh::KDtree(meshs->vertices);
  xform xft;
  xform xfs = xform(common::Matrix4fFromMatrix3d(T611.matrix()).data());
  cout << xfs << endl;
  vector<float> weights1, weights2;
  ICP(mesht, meshs, xft, xfs, kdt, kds, weights1, weights2, 0.0f, 2, ICP_RIGID);
  cout << xfs << endl;
}
