// Bunny, Convergence Rate

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

double Err(const vector<Vector3d> &pc, const Affine3d &aff) {
  double sum = 0.;
  auto pc2 = TransformPoints(pc, aff);
  for (size_t i = 0; i < pc.size(); ++i) sum += (pc2[i] - pc[i]).norm();
  return sum / pc.size();
}

double Err(const vector<Vector3d> &tgt,
           const vector<Vector3d> &src,
           const Affine3d &aff) {
  double sum = 0.;
  auto src2 = TransformPoints(src, aff);
  for (size_t i = 0; i < tgt.size(); ++i) sum += (src2[i] - tgt[i]).norm();
  return sum / tgt.size();
}

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

int main() {
  PointCloudType::Ptr pc = PointCloudType::Ptr(new PointCloudType);
  pcl::io::loadOBJFile<pcl::PointXYZ>(
      JoinPath(WSPATH, "src/ndt/data/bunny.obj"), *pc);
  for (auto &pt : *pc) pt.x *= 10., pt.y *= 10., pt.z *= 10.;
  vector<Vector3d> tgt;
  for (const auto &pt : *pc) tgt.push_back(Vector3d(pt.x, pt.y, pt.z));

  double tgterr = 1.;
  auto dirs = GenerateEvenPointsOnSphere(1);
  default_random_engine dre;
  auto urd = uniform_real_distribution<>();
  auto uid = uniform_int_distribution<>(0, 100);

  int trials = 1000;
  vector<vector<double>> r1, r2;
  int cnt1 = -1, cnt2 = -1;
  for (int i = 0; i < trials; ++i) {
    auto tdir = dirs[uid(dre)];
    auto rdir = dirs[uid(dre)];
    double tnum = urd(dre);
    double rnum = urd(dre);
    double r = 0.;
    int it = 0;
    Affine3d aff;
    for (; it < 10 && (r < 0.9999 || r > 1.0001); ++it) {
      aff = Translation3d(tnum * tdir) * AngleAxisd(Deg2Rad(rnum), rdir);
      float err = Err(tgt, aff);
      r = tgterr / err;
      tnum *= r;
      rnum *= r;
    }
    auto src = TransformPoints(tgt, aff);

    auto m1 = NDTMatcher::GetBasic({kNDT, k1to1, kAnalytic, kNoReject}, 0.5);
    m1.SetSource(src);
    m1.SetTarget(tgt);
    m1.Align();
    cnt1 = max(cnt1, (int)m1.tfs().size() - 1);
    r1.push_back({});
    for (size_t j = 1; j < m1.tfs().size() - 1u; ++j)
      r1.back().push_back(Err(tgt, src, m1.tfs()[j]));

    auto m2 = NDTMatcher::GetBasic({kNNDT, k1to1, kAnalytic, kNoReject}, 0.5);
    m2.SetSource(src);
    m2.SetTarget(tgt);
    m2.Align();
    cnt2 = max(cnt2, (int)m2.tfs().size() - 1);
    r2.push_back({});
    for (size_t j = 1; j < m2.tfs().size() - 1u; ++j)
      r2.back().push_back(Err(tgt, src, m2.tfs()[j]));
  }

  cout << "ndt = [";
  for (int i = 0; i < cnt1 - 1; ++i) {
    cout << "[";
    for (size_t j = 0; j < r1.size(); ++j) {
      if (i >= (int)r1[j].size())
        cout << r1[j].back() << ", ";
      else
        cout << r1[j][i] << ", ";
    }
    cout << "],\n";
  }
  cout << "]\n";

  cout << "nvndt = [";
  for (int i = 0; i < cnt2 - 1; ++i) {
    cout << "[";
    for (size_t j = 0; j < r2.size(); ++j) {
      if (i >= (int)r2[j].size())
        cout << r2[j].back() << ", ";
      else
        cout << r2[j][i] << ", ";
    }
    cout << "],\n";
  }
  cout << "]\n";
}
