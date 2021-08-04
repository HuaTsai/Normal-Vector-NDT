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
#include <common/EgoPointClouds.h>
#include <common/common.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32.h>

#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.hpp>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;

geometry_msgs::PoseStamped MakePST(const ros::Time &time, const Matrix4f &mtx) {
  geometry_msgs::PoseStamped ret;
  ret.header.frame_id = "map";
  ret.header.stamp = time;
  ret.pose = tf2::toMsg(common::Affine3dFromMatrix4f(mtx));
  return ret;
}

int main(int argc, char **argv) {
  double huber;
  int frames, method;
  double cell_size, radius, rvar, tvar;
  string infile, outfile;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("infile,i", po::value<string>(&infile)->required(), "Input data path")
      ("outfile,o", po::value<string>(&outfile)->required(), "Output file path")
      ("method,m", po::value<int>(&method)->default_value(0), "Output file path")
      ("frames,f", po::value<int>(&frames)->default_value(5), "Frames")
      ("rvar", po::value<double>(&rvar)->default_value(0.0625), "Intrinsic radius variance")
      ("tvar", po::value<double>(&tvar)->default_value(0.0001), "Intrinsic theta variance")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(1.5), "Radius")
      ("huber,u", po::value<double>(&huber)->default_value(5.0), "Use Huber loss");
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
  SerializationInput(infile, vepcs);
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

    if (method == 0) {
      // SNDT Method
      SNDTParameters params;
      auto datat = Augment(vepcs, i, i + f - 1, Tio, Tios);
      auto mapt = MakeSNDTMap(datat, params);

      auto datas = Augment(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);
      auto maps = MakeSNDTMap(datas, params);

      auto T = SNDTMatch(mapt, maps, params, Tio);

      cout << "Run SNDT Matching (" << i << "/" << n - 2 * f << ")" << endl;

      Tr = Tr * common::Matrix4fFromMatrix3d(T.matrix());
    } else if (method == 1) {
      // NDTD2D method
      NDTD2DParameters params;
      auto datat = Augment(vepcs, i, i + f - 1, Tio, Tios);
      auto mapt = MakeNDTMap(datat, params);

      auto datas = Augment(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);
      auto maps = MakeNDTMap(datas, params);

      auto T = NDTD2DMatch(mapt, maps, params, Tio);

      cout << "Run NDTD2D Matching (" << i << "/" << n - 2 * f << ")" << endl;

      Tr = Tr * common::Matrix4fFromMatrix3d(T.matrix());
    } else if (method == 2) {
      // SICP method
      vector<Vector2d> tgt = AugmentPoints(vepcs, i, i + f - 1, Tio, Tios);
      vector<Vector2d> src = AugmentPoints(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);

      SICPParameters params;
      auto T = SICPMatch(tgt, src, params, Tio);

      cout << "Run SICP Matching (" << i << "/" << n - 2 * f << ")" << endl;

      Tr = Tr * common::Matrix4fFromMatrix3d(T.matrix());
    }

    vp.push_back(MakePST(vepcs[i + f].stamp, Tr));
  } 

  nav_msgs::Path path;
  path.header.frame_id = "map";
  path.header.stamp = vp[0].header.stamp;
  path.poses = vp;
  SerializationOutput(outfile, path);
}
