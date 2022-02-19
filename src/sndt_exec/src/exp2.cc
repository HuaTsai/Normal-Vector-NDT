// NDT, SNDT decisions of d2
#include <bits/stdc++.h>
#include <common/common.h>
#include <metric/metric.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

double Dist(const nav_msgs::Path &gt, ros::Time t1, ros::Time t2) {
  auto p1 = GetPose(gt.poses, t1);
  auto p2 = GetPose(gt.poses, t2);
  return Vector2d(p1.position.x - p2.position.x, p1.position.y - p2.position.y)
      .norm();
}

Affine2d GetBenchMark(const nav_msgs::Path &gtpath,
                      const ros::Time &t1,
                      const ros::Time &t2) {
  Affine3d To, Ti;
  tf2::fromMsg(GetPose(gtpath.poses, t2), To);
  tf2::fromMsg(GetPose(gtpath.poses, t1), Ti);
  Affine3d Tgt3 = Conserve2DFromAffine3d(Ti.inverse() * To);
  Affine2d ret = Translation2d(Tgt3.translation()(0), Tgt3.translation()(1)) *
                 Rotation2Dd(Tgt3.rotation().block<2, 2>(0, 0));
  return ret;
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  string d;
  double c, r, v;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("h,h", "Produce help message")
      ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
      ("v,v", po::value<double>(&v)->default_value(0), "Voxel")
      ("c,c", po::value<double>(&c)->default_value(1.5), "Cell Size")
      ("r,r", po::value<double>(&r)->required(), "Meters to Next Match");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  nav_msgs::Path gtpath;
  SerializationInput(JoinPath(GetDataPath(d), "gt.ser"), gtpath);
  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(d), "lidar.ser"), vpc);
  int n = vpc.size() - 1;

  cout << "  all n = " << n << endl;
  vector<int> ids{0};
  for (int i = 0, j = 1; i < n && j < n; ++j) {
    if (Dist(gtpath, vpc[i].header.stamp, vpc[j].header.stamp) < r) continue;
    ids.push_back(j);
    i = j;
  }
  cout << "valid n = " << ids.size() << endl;

  vector<pair<int, double>> res;
  for (int i = 0; i < ids.size() - 1; ++i) {
    auto tgt = PCMsgTo2D(vpc[ids[i]], v);
    auto src = PCMsgTo2D(vpc[ids[i + 1]], v);
    auto tj = vpc[ids[i + 1]].header.stamp;
    auto Tgt = GetBenchMark(gtpath, vpc[ids[i]].header.stamp,
                            vpc[ids[i + 1]].header.stamp);

    vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

    Stat *len = nullptr;
    int bestid = 0;
    double bestr = numeric_limits<double>::max(), bestt = 0;
    for (int d2 : {0, 1, 2, 3, 4, 5, 6}) {
      D2DNDTParameters params5;
      params5.cell_size = c;
      params5.r_variance = params5.t_variance = 0;
      params5.d2 = pow(10, -d2);
      params5._usedtime.Start();
      params5.max_iterations = 0;
      params5.reject = false;
      auto tgt5 = MakeNDTMap(datat, params5);
      auto src5 = MakeNDTMap(datas, params5);
      auto T5 = D2DNDTMatch(tgt5, src5, params5);
      auto err5 = TransNormRotDegAbsFromAffine2d(Tgt.inverse() * T5);
      if (err5(0) < bestr) bestid = d2, bestr = err5(0), bestt = err5(1);
      
      if (!len) {
        vector<double> lens;
        int x = 0;
        for (auto corr : params5._corres[0]) {
          Vector2d u = src5[corr.first]->GetPointMean() - tgt5[corr.second]->GetPointMean();
          Matrix2d c = src5[corr.first]->GetPointCov() + tgt5[corr.second]->GetPointCov();
          lens.push_back(u.dot(c.inverse() * u));
          ++x;
        }
        len = new Stat(lens);
      }
    }
    printf("%f: %.2f, %.2f -> %.2f ", pow(10, -bestid), bestr, bestt, len->mean + 3 * len->stdev);
    res.push_back({bestid, len->mean});
    len->PrintResult();
    delete len;
  }
  printf("x = np.array([");
  for (auto r : res)
    printf("%d, ", r.first);
  printf("])\n");
  printf("y = np.array([");
  for (auto r : res)
    printf("%f, ", r.second);
  printf("])\n");
}