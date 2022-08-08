
// Bunny @ Different tls & rots (ICP)
// TODO: may add more far angle
// TODO: more strict success theshold
#include <common/common.h>
#include <ndt/matcher.h>
#include <ndt/visuals.h>
#include <pcl/common/transforms.h>
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

int main() {
  PointCloudType::Ptr target_pcl = PointCloudType::Ptr(new PointCloudType);
  pcl::io::loadPCDFile<pcl::PointXYZ>(
      JoinPath(WSPATH, "src/ndt/data/bunny2.pcd"), *target_pcl);
  vector<Vector3d> tgt;
  for (const auto &pt : *target_pcl) tgt.push_back(Vector3d(pt.x, pt.y, pt.z));

  vector<double> rs = {0, 3, 6, 9, 12, 15};
  vector<double> as = {0, 20, 40, 60, 80};
  MatrixXd sc1(rs.size(), as.size()), sc2(rs.size(), as.size());

  int n = 20;
  tqdm bar;
  for (size_t i = 0; i < rs.size(); ++i) {
    for (size_t j = 0; j < as.size(); ++j) {
      bar.progress(i * rs.size() + j, rs.size() * as.size());
      auto affs = GenerateTFs(rs[i], as[j], n);
      int cnt1 = 0, cnt2 = 0;
      for (auto aff : affs) {
        auto src = TransformPoints(tgt, aff);

        auto m1 = ICPMatcher::GetBasic({kNoReject});
        m1.SetSource(src);
        m1.SetTarget(tgt);
        auto res1 = m1.Align();
        auto err1 = TransNormRotDegAbsFromAffine3d(res1 * aff);
        if (err1(0) < 0.1 && err1(0) < 1) ++cnt1;

        auto m2 = SICPMatcher::GetBasic({kNoReject}, 0.5);
        m2.SetSource(src);
        m2.SetTarget(tgt);
        auto res2 = m2.Align();
        auto err2 = TransNormRotDegAbsFromAffine3d(res2 * aff);
        if (err2(0) < 0.1 && err2(0) < 1) ++cnt2;
      }
      sc1(i, j) = double(cnt1) / (n * n) * 100.;
      sc2(i, j) = double(cnt2) / (n * n) * 100.;
    }
  }
  bar.finish();

  cout << sc1 << endl << endl;
  cout << sc2 << endl;
}
