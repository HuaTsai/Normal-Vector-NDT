// Bunny: NDT
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

struct Res {
  Res()
      : te(0),
        re(0),
        iter(0),
        bud(0),
        nm(0),
        ndt(0),
        opt(0),
        oth(0),
        ttl(0),
        cnt(0) {}
  void Show() {
    printf(
        "cnt: %d, iter: %.2f, bud: %.2f, nm: %.20f, ndt: %.2f, opt: %.2f, oth: "
        "%.2f, ttl: %.2f\n",
        cnt, iter / cnt, bud / cnt, nm / cnt, ndt / cnt, opt / cnt, oth / cnt,
        ttl / cnt);
  }
  double te, re;
  double iter;
  double bud, nm, ndt, opt, oth, ttl;
  int cnt;
};

void Update(const Affine3d &dtf, const Matcher &m, Res &r) {
  double te = TransNormRotDegAbsFromAffine3d(dtf)(0);
  double re = TransNormRotDegAbsFromAffine3d(dtf)(1);
  if (te < 0.1 && re < 1) {
    ++r.cnt;
    r.te += te;
    r.re += re;
    r.iter += m.iteration();
    r.bud += m.timer().build() / 1000.;
    r.nm += m.timer().normal() / 1000.;
    r.ndt += m.timer().ndt() / 1000.;
    r.opt += m.timer().optimize() / 1000.;
    r.oth += m.timer().others() / 1000.;
    r.ttl += m.timer().total() / 1000.;
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
  auto tfs = GenerateTFs(1, 10, 20);
  Res r1, r2;
  for (auto tf : tfs) {
    auto src2 = TransformPoints(src, tf);

    auto m1 = NDTMatcher::GetBasic({kNDT, k1to1, kAnalytic, kNoReject}, cs);
    m1.SetSource(src2);
    m1.SetTarget(tgt);
    auto res1 = m1.Align();
    Update(res1 * tf, m1, r1);

    auto m2 = NDTMatcher::GetBasic({kNVNDT, k1to1, kAnalytic, kNoReject}, cs);
    m2.SetSource(src2);
    m2.SetTarget(tgt);
    auto res2 = m2.Align();
    Update(res2 * tf, m2, r2);
  }
  r1.Show();
  r2.Show();
}
