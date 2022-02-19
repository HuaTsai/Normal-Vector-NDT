#include <bits/stdc++.h>
#include <common/common.h>
#include <metric/metric.h>
#include <sndt/visuals.h>
#include <sndt_exec/wrapper.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>
#include <sndt/visuals.h>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

int main() {
  vector<double> pxs{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,
                     10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
                     20., 21., 22., 23., 24., 25., 26., 27., 28., 29.};
  vector<double> pys{
      0.,          0.09588511,  0.33658839,  0.59849699,  0.72743794,
      0.59847214,  0.16934401,  -0.49109652, -1.21088399, -1.75955421,
      -1.91784855, -1.55218872, -0.6705972,  0.55931197,  1.83956248,
      2.81399993,  3.16594639,  2.71485618,  1.48362655,  -0.28557426,
      -2.17608444, -3.69472219, -4.39995691, -4.02708,    -2.57555001,
      -0.33160949, 2.18486859,  4.3404359,   5.54740119,  5.42239132};

  vector<double> qxs{
      -2.,         -1.36069423, -0.82379037, -0.30188094, 0.31405082,
      1.11235019,  2.12289639,  3.29700515,  4.51307853,  5.60815375,
      6.42719153,  6.87573776,  6.9594652,   6.79689487,  6.59872784,
      6.61680328,  7.07504634,  8.10112206,  9.67883967,  11.63696034,
      13.68085969, 15.46180552, 16.66758855, 17.11103155, 16.79175162,
      15.91215285, 14.83984091, 14.02273143, 13.87638487, 14.67188698};
  vector<double> qys{
      5.,          5.77490779,  6.6522175,   7.54452163,  8.34280343,
      8.95871762,  9.36238498,  9.60248979,  9.80062997,  10.11976832,
      10.7149441,  11.68061143, 13.01109755, 14.58788144, 16.20026204,
      17.59640015, 18.55237066, 18.9405085,  18.77700445, 18.23309735,
      17.60341156, 17.23667929, 17.44510982, 18.41588039, 20.14937387,
      22.44318621, 24.92971171, 27.16103475, 28.72159487, 29.34030633};

  vector<Vector2d> p, q;
  for (size_t i = 0; i < pxs.size(); ++i) {
    p.push_back(Vector2d(pxs[i], pys[i]));
    q.push_back(Vector2d(qxs[i], qys[i]));
  }

  vector<pair<vector<Vector2d>, Affine2d>> datas{{p, Affine2d::Identity()}};
  vector<pair<vector<Vector2d>, Affine2d>> datat{{q, Affine2d::Identity()}};


  // auto np = ComputeNormals(p, 2);
  // for (int i = 0; i < np.size(); ++i) {
  //   cout << i << ": " << np[i].transpose() << endl;
  // }


  ICPParameters params1;
  // params1.reject = true;
  params1._usedtime.Start();
  auto tgt1 = MakePoints(datat, params1);
  auto src1 = MakePoints(datas, params1);
  auto T1 = ICPMatch(tgt1, src1, params1);
  cout << params1._ceres_iteration << endl;
  params1._usedtime.Show();
  cout << T1.matrix() << endl;

  Pt2plICPParameters params2;
  params2.radius = 2;
  params2._usedtime.Start();
  auto tgt2 = MakePoints(datat, params2);
  auto src2 = MakePoints(datas, params2);
  auto T2 = Pt2plICPMatch(tgt2, src2, params2);
  cout << params2._ceres_iteration << endl;
  params2._usedtime.Show();
  cout << T2.matrix() << endl;

  SICPParameters params3;
  params3.radius = 2;
  params3._usedtime.Start();
  auto tgt3 = MakePoints(datat, params3);
  auto src3 = MakePoints(datas, params3);
  auto T3 = SICPMatch(tgt3, src3, params3);
  cout << params3._ceres_iteration << endl;
  params3._usedtime.Show();
  cout << T3.matrix() << endl;

  // TODO: print to rviz

  // if (ms.count(5)) {
  //   D2DNDTParameters params5;
  //   params5.cell_size = cell_size;
  //   params5.r_variance = params5.t_variance = 0;
  //   params5.d2 = d2;
  //   params5._usedtime.Start();
  //   auto tgt5 = MakeNDTMap(datat, params5);
  //   auto src5 = MakeNDTMap(datas, params5);
  //   auto T5 = D2DNDTMatch(tgt5, src5, params5);
  // }

  // if (ms.count(7)) {
  //   D2DNDTParameters params7;
  //   params7.cell_size = cell_size;
  //   params7.r_variance = params7.t_variance = 0;
  //   params7.d2 = d2;
  //   params7._usedtime.Start();
  //   auto tgt7 = MakeNDTMap(datat, params7);
  //   auto src7 = MakeNDTMap(datas, params7);
  //   auto T7 = SNDTMatch2(tgt7, src7, params7);
  // }
}