/**
 * @file test_path.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief run for whole data log24, log62, log62-2
 * @version 0.1
 * @date 2021-07-17
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <bits/stdc++.h>
#include <sndt_exec/wrapper.hpp>
#include <boost/program_options.hpp>
#include <common/EgoPointClouds.h>
#include <sndt/ndt_visualizations.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Float64MultiArray.h>
#include <nav_msgs/Path.h>

using namespace std;
using namespace visualization_msgs;
namespace po = boost::program_options;

// start, ..., end, end+1
// <<------- T -------->>
vector<pair<MatrixXd, Affine2d>> Augment(
    const vector<common::EgoPointClouds> &vepcs, int start, int end,
    Affine2d &T, vector<Affine2d> &allT) {
  vector<pair<MatrixXd, Affine2d>> ret;
  double dx = 0, dy = 0, dth = 0;
  for (int i = start; i <= end; ++i) {
    Affine2d T0i = Rotation2Dd(dth) * Translation2d(dx, dy);
    allT.push_back(T0i);
    for (const auto &pc : vepcs[i].pcs) {
      MatrixXd fi(2, pc.points.size());
      for (int i = 0; i < fi.cols(); ++i)
        fi.col(i) = Vector2d(pc.points[i].x, pc.points[i].y);
      Affine3d aff;
      tf2::fromMsg(pc.origin, aff);
      Matrix3d mtx = Matrix3d::Identity();
      mtx.block<2, 2>(0, 0) = aff.matrix().block<2, 2>(0, 0);
      mtx.block<2, 1>(0, 2) = aff.matrix().block<2, 1>(0, 3);
      ret.push_back(make_pair(fi, T0i * Affine2d(mtx)));
    }
    double dt = (vepcs[i + 1].stamp - vepcs[i].stamp).toSec();
    dx += vepcs[i].vxyt[0] * dt;
    dy += vepcs[i].vxyt[1] * dt;
    dth += vepcs[i].vxyt[2] * dt;
  }
  T = Rotation2Dd(dth) * Translation2d(dx, dy);
  return ret;
}

geometry_msgs::PoseStamped MakePST(const ros::Time &time, const Matrix4f &mtx) {
  geometry_msgs::PoseStamped ret;
  ret.header.frame_id = "map";
  ret.header.stamp = time;
  ret.pose = tf2::toMsg(common::Affine3dFromMatrix4f(mtx));
  return ret;
}

int main(int argc, char **argv) {
  bool usehuber;
  int frames;
  double cell_size, radius, rvar, tvar;
  string infile, outfile;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("infile,i", po::value<string>(&infile)->required(), "Input data path")
      ("outfile,o", po::value<string>(&outfile)->required(), "Output file path")
      ("frames,f", po::value<int>(&frames)->default_value(5), "Frames")
      ("rvar", po::value<double>(&rvar)->default_value(0.0625), "Intrinsic radius variance")
      ("tvar", po::value<double>(&rvar)->default_value(0.0001), "Intrinsic theta variance")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(1.5), "Radius")
      ("usehuber,u", po::value<bool>(&usehuber)->default_value(false)->implicit_value(true), "Use Huber loss");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  ros::init(argc, argv, "test_path");
  ros::NodeHandle nh;
  vector<common::EgoPointClouds> vepcs;
  common::SerializationInput(infile, vepcs);
  int n = vepcs.size() / frames * frames;
  Matrix4f Tr = Matrix4f::Identity();
  vector<geometry_msgs::PoseStamped> vp;
  vp.push_back(MakePST(vepcs[0].stamp, Tr));
  // i-f, ..., i-1 | i, i+1, ..., i+f-1 | i+f, ..., i+2f-1 | i+2f -> actual id
  // ..., ...,  m  | i, ..., ...,   n   |  o , ...,   p    |  q   -> symbol id
  // -- frames  -- | ----- target ----- | ---- source ---- |
  //                 <<---- Computed T ---->>
  //                Tr (before iter)      Tr (after iter)
  int f = frames;
  for (int i = 0; i < n - f; i += f) {
    Affine2d Tio, Toq;
    vector<Affine2d> Tios, Toqs;

    auto datat = Augment(vepcs, i, i + f - 1, Tio, Tios);
    auto mapt = MakeMap(datat, {rvar, tvar}, {cell_size, radius});
    int tvc = count_if(mapt.begin(), mapt.end(), [](NDTCell *cell) { return cell->BothHasGaussian(); });

    auto datas = Augment(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);
    auto maps = MakeMap(datas, {rvar, tvar}, {cell_size, radius});
    int svc = count_if(maps.begin(), maps.end(), [](NDTCell *cell) { return cell->BothHasGaussian(); });

    NDTMatcher matcher;
    matcher.verbose = true;
    matcher.SetStrategy(NDTMatcher::kUSE_CELLS_GREATER_THAN_TWO_POINTS);
    matcher.usehuber = usehuber;
    auto T = matcher.CeresMatch(mapt, maps, Tio);

    cout << "Run Matching (" << i << "/" << n - 2 * f << "): "
         << common::XYTDegreeFromMatrix3d(Tio.matrix()).transpose()
         << " -> " << common::XYTDegreeFromMatrix3d(T).transpose()
         << "(" << tvc << ", " << svc << ")" << endl;

    Tr = Tr * common::Matrix4fFromMatrix3d(T);
    cout << Tr << endl;
    vp.push_back(MakePST(vepcs[i + f].stamp, Tr));
  } 

  nav_msgs::Path path;
  path.header.frame_id = "map";
  path.header.stamp = vp[0].header.stamp;
  path.poses = vp;
  common::SerializationOutput(outfile, path);
}
