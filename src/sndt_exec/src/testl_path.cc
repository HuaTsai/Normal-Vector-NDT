#include <ros/ros.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <sensor_msgs/CompressedImage.h>
#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <rosbag/bag.h>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;

vector<Vector2d> PCMsgTo2D(const sensor_msgs::PointCloud2 &msg) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(msg, *pc);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr fpc(new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud(pc);
  vg.setLeafSize(1, 1, 1);
  vg.filter(*fpc);

  vector<Vector2d> ret;
  for (const auto &pt : *fpc)
    if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z))
      ret.push_back(Vector2d(pt.x, pt.y));
  return ret;
}

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
  Affine3d aff3 =
      Translation3d(0.943713, 0.000000, 1.840230) *
      Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 =
      Translation2d(aff3.translation()(0), aff3.translation()(1)) *
      Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  double huber, cell_size, radius;
  string data, outfolder;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("outfolder,o", po::value<string>(&outfolder)->required(), "Output file path")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(1.5), "Radius")
      ("huber,u", po::value<double>(&huber)->default_value(1.0), "Use Huber loss");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);
  nav_msgs::Path gtpath;
  SerializationInput(JoinPath(GetDataPath(data), "gt.ser"), gtpath);
  vector<sensor_msgs::CompressedImage> imb, imbl, imbr, imf, imfl, imfr;
  SerializationInput(JoinPath(GetDataPath(data), "back.ser"), imb);
  SerializationInput(JoinPath(GetDataPath(data), "back_left.ser"), imbl);
  SerializationInput(JoinPath(GetDataPath(data), "back_right.ser"), imbr);
  SerializationInput(JoinPath(GetDataPath(data), "front.ser"), imf);
  SerializationInput(JoinPath(GetDataPath(data), "front_left.ser"), imfl);
  SerializationInput(JoinPath(GetDataPath(data), "front_right.ser"), imfr);

  Matrix4f Tr1 = Matrix4f::Identity();
  Matrix4f Tr2 = Matrix4f::Identity();
  Matrix4f Tr3 = Matrix4f::Identity();

  vector<geometry_msgs::PoseStamped> vp1, vp2, vp3;
  vp1.push_back(MakePoseStampedMsg(vpc[0].header.stamp, Tr1));
  vp2.push_back(MakePoseStampedMsg(vpc[0].header.stamp, Tr2));
  vp3.push_back(MakePoseStampedMsg(vpc[0].header.stamp, Tr3));

  rosbag::Bag bag;
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

  // |  i-1  |   i   |  i+1  | -> actual id
  // |   m   |   i   |   n   | -> symbol id
  //         |- tgt -|- src -|
  //         <<----- T ----->>
  //           Tr(bf)  Tr(af)
  double avg1 = 0, avg2 = 0, avg3 = 0;
  int n = vpc.size();
  for (int i = 0; i < n - 1; ++i) {
    auto tgt = PCMsgTo2D(vpc[i]);
    auto src = PCMsgTo2D(vpc[i + 1]);
    vector<Vector2d> tgt3(tgt.size()), src3(src.size());
    transform(tgt.begin(), tgt.end(), tgt3.begin(), [&aff2](auto p) { return aff2 * p; });
    transform(src.begin(), src.end(), src3.begin(), [&aff2](auto p) { return aff2 * p; });
    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};
    auto srctime = vpc[i + 1].header.stamp;
    bag.write("source", srctime, EigenToPC(src3, srctime));
    // auto Tg = GetDiffT(vepcs, vpc[i].header.stamp, vpc[i + 1].header.stamp);
    auto Tg = Affine2d::Identity();

    // SNDT Method
    auto t1 = GetTime();
    SNDTParameters params1;
    params1.r_variance = params1.t_variance = 0;
    auto tgt1 = MakeSNDTMap(datat, params1);
    auto src1 = MakeSNDTMap(datas, params1);
    auto T1 = SNDTMatch(tgt1, src1, params1, Tg);
    Tr1 = Tr1 * Matrix4fFromMatrix3d(T1.matrix());
    vp1.push_back(MakePoseStampedMsg(vpc[i + 1].header.stamp, Tr1));
    vector<Vector2d> next1(tgt3.size());
    transform(tgt3.begin(), tgt3.end(), next1.begin(), [&T1](auto p) { return T1.inverse() * p; });
    bag.write("sndt", srctime, EigenToPC(next1, srctime));
    auto t2 = GetTime();
    avg1 += GetDiffTime(t1, t2);

    // NDT method
    auto t3 = GetTime();
    NDTParameters params2;
    params2.r_variance = params2.t_variance = 0;
    auto tgt2 = MakeNDTMap(datat, params2);
    auto src2 = MakeNDTMap(datas, params2);
    auto T2 = NDTD2DMatch(tgt2, src2, params2, Tg);
    Tr2 = Tr2 * Matrix4fFromMatrix3d(T2.matrix());
    vp2.push_back(MakePoseStampedMsg(vpc[i + 1].header.stamp, Tr2));
    vector<Vector2d> next2(tgt3.size());
    transform(tgt3.begin(), tgt3.end(), next2.begin(), [&T2](auto p) { return T2.inverse() * p; });
    bag.write("ndt", srctime, EigenToPC(next2, srctime));
    auto t4 = GetTime();
    avg2 += GetDiffTime(t3, t4);

    // SICP method
    auto t5 = GetTime();
    SICPParameters params3;
    auto T3 = SICPMatch(tgt3, src3, params3, Tg);
    Tr3 = Tr3 * Matrix4fFromMatrix3d(T3.matrix());
    vp3.push_back(MakePoseStampedMsg(vpc[i + 1].header.stamp, Tr3));
    vector<Vector2d> next3(tgt3.size());
    transform(tgt3.begin(), tgt3.end(), next3.begin(), [&T3](auto p) { return T3.inverse() * p; });
    bag.write("sicp", srctime, EigenToPC(next3, srctime));
    auto t6 = GetTime();
    avg3 += GetDiffTime(t5, t6);

    ::printf("\rRun SNDT/NDT/SICP Matching (%d/%d)", i, n - 2);
    fflush(stdout);
  }
  bag.close();
  avg1 /= (n - 1);
  avg2 /= (n - 1);
  avg3 /= (n - 1);
  ::printf("\nSNDT: %f, NDT: %f, SICP: %f\n", avg1, avg2, avg3);

  nav_msgs::Path estpath;
  estpath.header.frame_id = "map";
  estpath.header.stamp = vpc[0].header.stamp;

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
