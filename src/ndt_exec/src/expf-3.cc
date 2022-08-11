// Bunny: Error - Iter, Convergence RAte
#include <common/common.h>
#include <metric/metric.h>
#include <ndt/matcher.h>
#include <ndt/visuals.h>
#include <pcl/common/transforms.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>

using namespace std;
using namespace Eigen;
using PointCloudType = pcl::PointCloud<pcl::PointXYZ>;

vector<Vector3d> GenerateEvenPointsOnSphere(double radius, int samples = 100) {
  vector<Vector3d> ret;
  double phi = M_PI * (3. - std::sqrt(5.));
  for (int i = 0; i < samples; ++i) {
    double y = 1 - (i / double(samples - 1)) * 2;
    double r = std::sqrt(1 - y * y);
    double th = phi * i;
    double x = std::cos(th) * r;
    double z = std::sin(th) * r;
    ret.push_back(radius * Vector3d(x, y, z));
  }
  return ret;
}

vector<Affine3d> GenerateTFs(double r, double a, int n) {
  if (r < 0 || a < 0) std::exit(1);
  auto tls = GenerateEvenPointsOnSphere(r, n);
  auto axes = GenerateEvenPointsOnSphere(1, n);
  vector<Affine3d> ret;
  for (const auto &tl : tls) {
    for (const auto &axis : axes) {
      ret.push_back(Translation3d(tl) * AngleAxisd(Deg2Rad(a), axis));
    }
  }
  return ret;
}

void Check(const Affine3d &dtf,
           const vector<Affine3d> &tfs,
           const vector<Vector3d> &src,
           const vector<Vector3d> &tgt) {
  double te = TransNormRotDegAbsFromAffine3d(dtf)(0);
  double re = TransNormRotDegAbsFromAffine3d(dtf)(1);
  cout << te << ", " << re << endl;
  for (auto tf : tfs) {
    double err = 0;
    for (size_t i = 0; i < src.size(); ++i)
      err += (tf * src[i] - tgt[i]).squaredNorm();
    cout << sqrt(err / src.size()) << ", ";
  }
}

int main() {
  PointCloudType::Ptr source_pcl = PointCloudType::Ptr(new PointCloudType);
  PointCloudType::Ptr target_pcl = PointCloudType::Ptr(new PointCloudType);
  pcl::io::loadOBJFile<pcl::PointXYZ>(
      JoinPath(WSPATH, "src/ndt/data/bunny.obj"), *source_pcl);
  pcl::io::loadOBJFile<pcl::PointXYZ>(
      JoinPath(WSPATH, "src/ndt/data/bunny.obj"), *target_pcl);
  for (auto &pt : *source_pcl) pt.x *= 10., pt.y *= 10., pt.z *= 10.;
  for (auto &pt : *target_pcl) pt.x *= 10., pt.y *= 10., pt.z *= 10.;
  vector<Vector3d> src, tgt;
  for (const auto &pt : *source_pcl) src.push_back(Vector3d(pt.x, pt.y, pt.z));
  for (const auto &pt : *target_pcl) tgt.push_back(Vector3d(pt.x, pt.y, pt.z));

  double cs = 0.5;
  auto tfs = GenerateTFs(1, 10, 10);
  for (auto tf : tfs) {
    auto src2 = TransformPoints(src, tf);
    double err = 0;
    for (size_t i = 0; i < src2.size(); ++i)
      err += (tf * src2[i] - tgt[i]).squaredNorm();
    cout << sqrt(err / src2.size()) << ", ";
  }
  cout << endl;

  auto tf = tfs[50];
  auto src2 = TransformPoints(src, tf);

  auto m1 = NDTMatcher::GetBasic({kNDT, k1to1, kAnalytic, kNoReject}, cs);
  m1.SetSource(src2);
  m1.SetTarget(tgt);
  auto res1 = m1.Align();
  Check(res1 * tf, m1.tfs(), src2, tgt);

  auto m2 = NDTMatcher::GetBasic({kNVNDT, k1to1, kAnalytic, kNoReject}, cs);
  m2.SetSource(src2);
  m2.SetTarget(tgt);
  auto res2 = m2.Align();
  Check(res2 * tf, m2.tfs(), src2, tgt);

  auto m3 = ICPMatcher::GetBasic({kNoReject});
  m3.SetSource(src2);
  m3.SetTarget(tgt);
  auto res3 = m3.Align();
  Check(res3 * tf, m3.tfs(), src2, tgt);

  auto m4 = SICPMatcher::GetBasic({kNoReject}, 0.5);
  m4.SetSource(src2);
  m4.SetTarget(tgt);
  auto res4 = m4.Align();
  Check(res4 * tf, m4.tfs(), src2, tgt);
}
