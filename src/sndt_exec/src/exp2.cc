// finds what works well and what works bad
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

// TODO
int main(int argc, char **argv) {
  // Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
  //                 Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  // aff3 = Conserve2DFromAffine3d(aff3);
  // Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
  //                 Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  // string d;
  // double c, d2, v;
  // po::options_description desc("Allowed options");
  // // clang-format off
  // desc.add_options()
  //     ("h,h", "Produce help message")
  //     ("d,d", po::value<string>(&d)->required(), "Data (logxx)")
  //     ("v,v", po::value<double>(&v)->default_value(0), "Voxel")
  //     ("c,c", po::value<double>(&c)->default_value(1.5), "Cell Size")
  //     ("d2", po::value<double>(&d2)->required(), "d2");
  // // clang-format on
  // po::variables_map vm;
  // po::store(po::parse_command_line(argc, argv, desc), vm);
  // if (vm.count("help")) {
  //   cout << desc << endl;
  //   return 1;
  // }
  // po::notify(vm);

  // nav_msgs::Path gtpath;
  // SerializationInput(JoinPath(GetDataPath(d), "gt.ser"), gtpath);
  // vector<sensor_msgs::PointCloud2> vpc;
  // SerializationInput(JoinPath(GetDataPath(d), "lidar.ser"), vpc);
  // int n = vpc.size() - 1;

  // vector<int> ids{0};
  // for (int i = 0, j = 1; i < n && j < n; ++j) {
  //   if (Dist(gtpath, vpc[i].header.stamp, vpc[j].header.stamp) < r) continue;
  //   ids.push_back(j);
  //   i = j;
  // }
  // cout << n << " -> " << ids.size() << endl;


  // auto t0 = vpc[0].header.stamp;
  // Res r5, r7;
  // r5.path = InitFirstPose(t0);
  // r7.path = InitFirstPose(t0);
  // vector<double> e5t, e5r, e7t, e7r;

  // tqdm bar;
  // for (size_t i = 0; i < ids.size() - 1; ++i) {
  //   bar.progress(i, ids.size());
  //   auto tgt = PCMsgTo2D(vpc[ids[i]], v);
  //   auto src = PCMsgTo2D(vpc[ids[i + 1]], v);
  //   auto tj = vpc[ids[i + 1]].header.stamp;

  //   vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, aff2}};
  //   vector<pair<vector<Vector2d>, Affine2d>> datas{{src, aff2}};

  //     D2DNDTParameters params5;
  //     params5.reject = true;
  //     params5.cell_size = c;
  //     params5.d2 = d2;
  //     params5._usedtime.Start();
  //     auto tgt5 = MakeNDT(datat, params5);
  //     auto src5 = MakeNDT(datas, params5);
  //     Affine2d T5;
  //     if (tr)
  //       T5 = DMatch(tgt5, src5, params5);
  //     else
  //       T5 = D2DNDTMatch(tgt5, src5, params5);
  //     e5t.push_back(TransNormRotDegAbsFromAffine2d(T5)(0));
  //     e5r.push_back(TransNormRotDegAbsFromAffine2d(T5)(1));
  //     Updates(params5, r5, tj, T5);

  //     D2DNDTParameters params7;
  //     params7.reject = true;
  //     params7.cell_size = c;
  //     params7.d2 = d2;
  //     params7._usedtime.Start();
  //     auto tgt7 = MakeNDT(datat, params7);
  //     auto src7 = MakeNDT(datas, params7);
  //     Affine2d T7;
  //     if (tr)
  //       T7 = SMatch(tgt7, src7, params7);
  //     else
  //       T7 = SNDTMatch2(tgt7, src7, params7);
  //     e7t.push_back(TransNormRotDegAbsFromAffine2d(T7)(0));
  //     e7r.push_back(TransNormRotDegAbsFromAffine2d(T7)(1));
  //     Updates(params7, r7, tj, T7);
  //   }
  // }
  // bar.finish();
  // Stat(e5t).PrintResult();
  // Stat(e5r).PrintResult();
  // Stat(e7t).PrintResult();
  // Stat(e7r).PrintResult();
  // PrintTime(r5, r7);

  // ros::init(argc, argv, "exp2");
  // ros::NodeHandle nh;
  // auto pub5 = nh.advertise<nav_msgs::Path>("path5", 0, true);
  // auto pub7 = nh.advertise<nav_msgs::Path>("path7", 0, true);
  // auto pubgt = nh.advertise<nav_msgs::Path>("pathg", 0, true);

  // MakeGtLocal(gtpath, t0);
  // pub5.publish(r5.path);
  // pub7.publish(r7.path);
  // pubgt.publish(gtpath);
  // PrintResult(r5.path, gtpath);
  // PrintResult(r7.path, gtpath);
  // r5.Print();
  // r7.Print();
  // ros::spin();
}

