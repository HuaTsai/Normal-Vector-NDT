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
#include <sensor_msgs/CompressedImage.h>
#include <rosbag/bag.h>
#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.hpp>
#include <pcl_ros/point_cloud.h>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;

sensor_msgs::PointCloud2 EigenToPC(const vector<Vector2d> &pts, const ros::Time &time) {
  pcl::PointCloud<pcl::PointXYZ> pc;
  for (auto pt : pts)
    pc.push_back(pcl::PointXYZ(pt(0), pt(1), 0));
  sensor_msgs::PointCloud2 ret;
  pcl::toROSMsg(pc, ret);
  ret.header.frame_id = "map";
  ret.header.stamp = time;
  return ret;
}

int main(int argc, char **argv) {
  bool bagout;
  double huber, cell_size, radius, rvar, tvar;
  int frames;
  string data, outfolder;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("outfolder,o", po::value<string>(&outfolder)->required(), "Output file path")
      ("frames,f", po::value<int>(&frames)->default_value(5), "Frames")
      ("rvar", po::value<double>(&rvar)->default_value(0.0625), "Intrinsic radius variance")
      ("tvar", po::value<double>(&tvar)->default_value(0.0001), "Intrinsic theta variance")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(1.5), "Radius")
      ("huber,u", po::value<double>(&huber)->default_value(1.0), "Use Huber loss")
      ("bag,b", po::value<bool>(&bagout)->default_value(false)->implicit_value(true), "Write bag");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  vector<common::EgoPointClouds> vepcs;
  SerializationInput(JoinPath(GetDataPath(data), "vepcs.ser"), vepcs);
  nav_msgs::Path gtpath;
  SerializationInput(JoinPath(GetDataPath(data), "gt.ser"), gtpath);
  vector<sensor_msgs::CompressedImage> imb, imbl, imbr, imf, imfl, imfr;
  if (bagout) {
    SerializationInput(JoinPath(GetDataPath(data), "back.ser"), imb);
    SerializationInput(JoinPath(GetDataPath(data), "back_left.ser"), imbl);
    SerializationInput(JoinPath(GetDataPath(data), "back_right.ser"), imbr);
    SerializationInput(JoinPath(GetDataPath(data), "front.ser"), imf);
    SerializationInput(JoinPath(GetDataPath(data), "front_left.ser"), imfl);
    SerializationInput(JoinPath(GetDataPath(data), "front_right.ser"), imfr);
  }

  Matrix4f Tr1 = Matrix4f::Identity();
  Matrix4f Tr2 = Matrix4f::Identity();
  Matrix4f Tr3 = Matrix4f::Identity();
  vector<geometry_msgs::PoseStamped> vp1, vp2, vp3;
  vp1.push_back(MakePoseStampedMsg(vepcs[0].stamp, Tr1));
  vp2.push_back(MakePoseStampedMsg(vepcs[0].stamp, Tr2));
  vp3.push_back(MakePoseStampedMsg(vepcs[0].stamp, Tr3));

  rosbag::Bag bag;
  if (bagout) {
    bag.open(JoinPath(outfolder, "replay" + data + ".bag"), rosbag::bagmode::Write);
    for (const auto &im : imb)
      bag.write("back/compressed", im.header.stamp, im);
    for (const auto &im : imbl)
      bag.write("back_left/compressed", im.header.stamp, im);
    for (const auto &im : imbr)
      bag.write("back_right/compressed", im.header.stamp, im);
    for (const auto &im : imf)
      bag.write("front/compressed", im.header.stamp, im);
    for (const auto &im : imfl)
      bag.write("front_left/compressed", im.header.stamp, im);
    for (const auto &im : imfr)
      bag.write("front_right/compressed", im.header.stamp, im);
  }

  // i-f, ..., i-1 | i, i+1, ..., i+f-1 | i+f, ..., i+2f-1 | i+2f -> actual id
  // ..., ...,  m  | i, ..., ...,   n   |  o , ...,   p    |  q   -> symbol id
  // -- frames  -- | ----- target ----- | ---- source ---- |
  //                 <<---- Computed T ---->>
  //                Tr (before iter)      Tr (after iter)
  double avg1 = 0, avg2 = 0, avg3 = 0;
  int n = vepcs.size() / frames * frames, f = frames;
  for (int i = 0; i < n - f; i += f) {
    Affine2d Tio, Toq;
    vector<Affine2d> Tios, Toqs;
    vector<Vector2d> tgt3 = AugmentPoints(vepcs, i, i + f - 1, Tio, Tios);
    vector<Vector2d> src3 = AugmentPoints(vepcs, i + f, i + 2 * f - 1, Toq, Toqs);
    auto srctime = vepcs[i + f].stamp;
    if (bagout)
      bag.write("source", srctime, EigenToPC(src3, srctime));

    // SNDT Method
    auto t1 = GetTime();
    SNDTParameters params1;
    auto tgt1 = MakeSNDTMap(Augment(vepcs, i, i + f - 1, Tio, Tios), params1);
    auto src1 = MakeSNDTMap(Augment(vepcs, i + f, i + 2 * f - 1, Toq, Toqs), params1);
    auto T1 = SNDTMatch(tgt1, src1, params1, Tio);
    Tr1 = Tr1 * Matrix4fFromMatrix3d(T1.matrix());
    vp1.push_back(MakePoseStampedMsg(vepcs[i + f].stamp, Tr1));
    vector<Vector2d> next1(tgt3.size());
    transform(tgt3.begin(), tgt3.end(), next1.begin(), [&T1](auto p) { return T1.inverse() * p; });
    if (bagout)
      bag.write("sndt", srctime, EigenToPC(next1, srctime));
    auto t2 = GetTime();
    avg1 += GetDiffTime(t1, t2);

    // NDTD2D method
    auto t3 = GetTime();
    D2DNDTParameters params2;
    auto tgt2 = MakeNDTMap(Augment(vepcs, i, i + f - 1, Tio, Tios), params2);
    auto src2 = MakeNDTMap(Augment(vepcs, i + f, i + 2 * f - 1, Toq, Toqs), params2);
    auto T2 = D2DNDTMatch(tgt2, src2, params2, Tio);
    Tr2 = Tr2 * Matrix4fFromMatrix3d(T2.matrix());
    vp2.push_back(MakePoseStampedMsg(vepcs[i + f].stamp, Tr2));
    vector<Vector2d> next2(tgt3.size());
    transform(tgt3.begin(), tgt3.end(), next2.begin(), [&T2](auto p) { return T2.inverse() * p; });
    if (bagout)
      bag.write("ndt", srctime, EigenToPC(next2, srctime));
    auto t4 = GetTime();
    avg2 += GetDiffTime(t3, t4);

    // SICP method
    auto t5 = GetTime();
    SICPParameters params3;
    auto T3 = SICPMatch(tgt3, src3, params3, Tio);
    Tr3 = Tr3 * Matrix4fFromMatrix3d(T3.matrix());
    vp3.push_back(MakePoseStampedMsg(vepcs[i + f].stamp, Tr3));
    vector<Vector2d> next3(tgt3.size());
    transform(tgt3.begin(), tgt3.end(), next3.begin(), [&T3](auto p) { return T3.inverse() * p; });
    if (bagout)
      bag.write("sicp", srctime, EigenToPC(next3, srctime));
    auto t6 = GetTime();
    avg3 += GetDiffTime(t5, t6);

    ::printf("\rRun SNDT/NDT/SICP Matching (%d/%d)", i, n - 2 * f);
    fflush(stdout);
  }
  if (bagout)
    bag.close();
  avg1 /= (n - 1);
  avg2 /= (n - 1);
  avg3 /= (n - 1);
  ::printf("\nSNDT: %f, NDT: %f, SICP: %f\n", avg1, avg2, avg3);

  nav_msgs::Path estpath;
  estpath.header.frame_id = "map";
  estpath.header.stamp = vepcs[0].stamp;

  auto est1 = estpath;
  est1.poses = vp1;
  auto gt1 = gtpath;
  MakeGtLocal(gt1, est1.poses[0].header.stamp);
  WriteToFile(est1, JoinPath(outfolder, "pc/sndt/pc_sndt_" + data, "stamped_traj_estimate.txt"));
  WriteToFile(gt1, JoinPath(outfolder, "pc/sndt/pc_sndt_" + data, "stamped_groundtruth.txt"));

  auto est2 = estpath;
  est2.poses = vp2;
  auto gt2 = gtpath;
  MakeGtLocal(gt2, est2.poses[0].header.stamp);
  WriteToFile(est2, JoinPath(outfolder, "pc/ndt/pc_ndt_" + data, "stamped_traj_estimate.txt"));
  WriteToFile(gt2, JoinPath(outfolder, "pc/ndt/pc_ndt_" + data, "stamped_groundtruth.txt"));

  auto est3 = estpath;
  est3.poses = vp3;
  auto gt3 = gtpath;
  MakeGtLocal(gt3, est3.poses[0].header.stamp);
  WriteToFile(est3, JoinPath(outfolder, "pc/sicp/pc_sicp_" + data, "stamped_traj_estimate.txt"));
  WriteToFile(gt3, JoinPath(outfolder, "pc/sicp/pc_sicp_" + data, "stamped_groundtruth.txt"));
}
