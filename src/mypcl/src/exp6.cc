// PCL Results
#include <common/common.h>
#include <metric/metric.h>
#include <mypcl/myd2dndt.h>
#include <nav_msgs/Path.h>
#include <pcl/conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <sensor_msgs/PointCloud2.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

void fromROS(const sensor_msgs::PointCloud2 &msg,
             pcl::PointCloud<pcl::PointXYZ>::Ptr pc) {
  pcl::PCLPointCloud2 pc2;
  pc2.data = msg.data;
  pc2.height = msg.height;
  pc2.width = msg.width;
  pc2.is_bigendian = msg.is_bigendian;
  pc2.point_step = msg.point_step;
  pc2.row_step = msg.row_step;
  pc2.is_dense = msg.is_dense;
  pc2.header.frame_id = msg.header.frame_id;
  pc2.header.seq = msg.header.seq;
  pc2.header.stamp = msg.header.stamp.toNSec() / 1000ull;
  pc2.fields.resize(msg.fields.size());
  for (size_t i = 0; i < msg.fields.size(); ++i) {
    pc2.fields[i].name = msg.fields[i].name;
    pc2.fields[i].offset = msg.fields[i].offset;
    pc2.fields[i].datatype = msg.fields[i].datatype;
    pc2.fields[i].count = msg.fields[i].count;
  }
  pcl::fromPCLPointCloud2(pc2, *pc);
}

void PCMsgTo3D(const sensor_msgs::PointCloud2 &msg,
               double voxel,
               pcl::PointCloud<pcl::PointXYZ>::Ptr pc) {
  fromROS(msg, pc);
  if (voxel != 0) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pc);
    vg.setLeafSize(voxel, voxel, voxel);
    vg.filter(*pc);
  }
}

nav_msgs::Path InitFirstPose(const ros::Time &time) {
  nav_msgs::Path path;
  path.header.frame_id = "map";
  path.header.stamp = time;
  path.poses.push_back(MakePoseStampedMsg(time, Eigen::Affine3d::Identity()));
  return path;
}

struct Res {
  Res() : Tr(Affine3d::Identity()) {}
  void Show() {
    printf(" its: %f / %f\n", Stat(its).mean, Stat(its).max);
    printf(" err: %f / %f\n", Stat(terr).rms, Stat(rerr).rms);
  }
  vector<double> its;
  vector<double> terr;
  vector<double> rerr;
  vector<double> opt;
  nav_msgs::Path path;
  Affine3d Tr;
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

void PrintRes(const nav_msgs::Path &est, const nav_msgs::Path &gt) {
  TrajectoryEvaluation te;
  te.set_estpath(est);
  te.set_gtpath(gt);
  te.set_evaltype(TrajectoryEvaluation::EvalType::kRelativeByLength);
  te.set_length(100);
  auto rpe = te.ComputeRMSError2D();
  cout << rpe.first.rms << " / " << rpe.second.rms << endl;
}

void PNumpy(vector<double> coll, string str) {
  cout << str << " = [";
  copy(coll.begin(), coll.end(), ostream_iterator<double>(cout, ", "));
  cout << "]\n";
}

void AddNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
               pcl::PointCloud<pcl::PointNormal>::Ptr out) {
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> est;
  est.setInputCloud(cloud);
  est.setSearchMethod(tree);
  // est.setKSearch(15);
  est.setRadiusSearch(1);
  est.compute(*normals);
  pcl::concatenateFields(*cloud, *normals, *out);
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  string d;
  int f, n;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
      ("f,f", po::value<int>(&f)->required()->default_value(1), "Frames")
      ("n,n", po::value<int>(&n)->default_value(-1), "Frames");
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
  Res r1, r2, r3, r4;
  r1.path = InitFirstPose(t0);
  r2.path = InitFirstPose(t0);
  r3.path = InitFirstPose(t0);
  r4.path = InitFirstPose(t0);
  std::vector<double> bent, benr;
  if (n == -1) n = vpc.size() - 1;
  for (int i = 0; i < n; i += f) {
    auto tj = vpc[i + f].header.stamp;
    auto ben = BM(gt, vpc[i].header.stamp, vpc[i + f].header.stamp);
    bent.push_back(TransNormRotDegAbsFromAffine3d(ben)(0));
    benr.push_back(TransNormRotDegAbsFromAffine3d(ben)(1));

    double voxel = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal>::Ptr srcn(
        new pcl::PointCloud<pcl::PointNormal>);
    PCMsgTo3D(vpc[i + f], voxel, src);
    pcl::transformPointCloud(*src, *src, aff3);
    AddNormal(src, srcn);

    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal>::Ptr tgtn(
        new pcl::PointCloud<pcl::PointNormal>);
    PCMsgTo3D(vpc[i], voxel, tgt);
    pcl::transformPointCloud(*tgt, *tgt, aff3);
    AddNormal(tgt, tgtn);

    pcl::PointCloud<pcl::PointXYZ> out;
    pcl::PointCloud<pcl::PointNormal> outn;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    auto t1 = GetTime();
    icp.align(out);
    auto t2 = GetTime();
    auto res1 = Affine3d(icp.getFinalTransformation().cast<double>());
    auto err1 = TransNormRotDegAbsFromAffine3d(res1 * ben.inverse());
    r1.terr.push_back(err1(0));
    r1.rerr.push_back(err1(1));
    r1.opt.push_back(GetDiffTime(t1, t2) / 1000.);
    r1.Tr = r1.Tr * res1;
    r1.path.poses.push_back(MakePoseStampedMsg(tj, r1.Tr));

    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setInputSource(src);
    ndt.setInputTarget(tgt);
    ndt.setResolution(1);
    auto t3 = GetTime();
    ndt.align(out);
    auto t4 = GetTime();
    auto res2 = Affine3d(ndt.getFinalTransformation().cast<double>());
    auto err2 = TransNormRotDegAbsFromAffine3d(res2 * ben.inverse());
    r2.terr.push_back(err2(0));
    r2.rerr.push_back(err2(1));
    r2.opt.push_back(GetDiffTime(t3, t4) / 1000.);
    r2.Tr = r2.Tr * res2;
    r2.path.poses.push_back(MakePoseStampedMsg(tj, r2.Tr));

    pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>
        sicp;
    sicp.setUseSymmetricObjective(true);
    sicp.setInputSource(srcn);
    sicp.setInputTarget(tgtn);
    auto t5 = GetTime();
    sicp.align(outn);
    auto t6 = GetTime();
    auto res3 = Affine3d(sicp.getFinalTransformation().cast<double>());
    auto err3 = TransNormRotDegAbsFromAffine3d(res3 * ben.inverse());
    r3.terr.push_back(err3(0));
    r3.rerr.push_back(err3(1));
    r3.opt.push_back(GetDiffTime(t5, t6) / 1000.);
    r3.Tr = r3.Tr * res3;
    r3.path.poses.push_back(MakePoseStampedMsg(tj, r3.Tr));

    pcl::NormalDistributionsTransformD2D<pcl::PointXYZ, pcl::PointXYZ> d2d;
    d2d.setInputSource(src);
    d2d.setInputTarget(tgt);
    d2d.setResolution(1);
    auto t7 = GetTime();
    d2d.align(out);
    auto t8 = GetTime();
    auto res4 = Affine3d(d2d.getFinalTransformation().cast<double>());
    auto err4 = TransNormRotDegAbsFromAffine3d(res4 * ben.inverse());
    r4.terr.push_back(err4(0));
    r4.rerr.push_back(err4(1));
    r4.opt.push_back(GetDiffTime(t7, t8) / 1000.);
    r4.Tr = r4.Tr * res4;
    r4.path.poses.push_back(MakePoseStampedMsg(tj, r4.Tr));
  }

  ros::init(argc, argv, "exp6");
  ros::NodeHandle nh;
  auto pub5 = nh.advertise<nav_msgs::Path>("path5", 0, true);
  auto pub7 = nh.advertise<nav_msgs::Path>("path7", 0, true);
  auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);
  pub5.publish(r1.path);
  pub7.publish(r2.path);
  pubgt.publish(gt);

  // PNumpy(r1.terr, "t1");
  // PNumpy(r1.rerr, "r1");
  // PNumpy(r1.opt, "opt1");
  // PNumpy(r2.terr, "t2");
  // PNumpy(r2.rerr, "r2");
  // PNumpy(r2.opt, "opt2");
  // PrintRes(r1.path, gt);
  // PrintRes(r2.path, gt);
  // PrintRes(r3.path, gt);
  // PrintRes(r4.path, gt);
  cout << "-------------" << endl;
  r1.Show();
  cout << "time average: " << Stat(r1.opt).mean << endl;
  cout << "-------------" << endl;
  r2.Show();
  cout << "time average: " << Stat(r2.opt).mean << endl;
  cout << "-------------" << endl;
  r3.Show();
  cout << "time average: " << Stat(r3.opt).mean << endl;
  cout << "-------------" << endl;
  r4.Show();
  cout << "time average: " << Stat(r4.opt).mean << endl;
  cout << endl;
  ros::spin();
}
