/**
 * @file test_0dis.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Evaluate Distribution of SICP and SNDT
 * @version 0.1
 * @date 2021-07-01
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <bits/stdc++.h>
#include <sndt_exec/wrapper.hpp>
#include <boost/program_options.hpp>
#include <common/EgoPointClouds.h>
#include <sndt/ndt_visualizations.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Float64MultiArray.h>

using namespace std;
using namespace visualization_msgs;
namespace po = boost::program_options;

vector<vector<double>> meancovs;
vector<MarkerArray> corres;
ros::Publisher pub7, pubmc;

struct CostFunctions {
  static vector<double> OldMeanAndCov(NDTCell *p, NDTCell *q) {
    Vector2d up = p->GetPointMean();
    Vector2d uq = q->GetPointMean();
    Vector2d unp = p->GetNormalMean();
    Vector2d unq = q->GetNormalMean();
    Matrix2d cp = p->GetPointCov();
    Matrix2d cq = q->GetPointCov();
    Matrix2d cnp = p->GetNormalCov();
    Matrix2d cnq = q->GetNormalCov();
    Vector2d m1 = up - uq;
    Vector2d m2 = unp + unq;
    Matrix2d c1 = cp + cq;
    Matrix2d c2 = cnp + cnq;
    return {m1.dot(m2), m1.dot(c2 * m1) + m2.dot(c1 * m2) + (c1 * c2).trace()};
  }

  static vector<double> NewMeanAndCov(NDTCell *p, NDTCell *q) {
    Vector2d up = p->GetPointMean();
    Vector2d uq = q->GetPointMean();
    Vector2d unp = p->GetNormalMean();
    Vector2d unq = q->GetNormalMean();
    Matrix2d cp = p->GetPointCov();
    Matrix2d cq = q->GetPointCov();
    // Matrix2d cnp = p->GetNormalCov();
    // Matrix2d cnq = q->GetNormalCov();
    Vector2d m1 = up - uq;
    Vector2d m2 = unp + unq;
    Matrix2d c1 = cp + cq;
    // Matrix2d c2 = cnp + cnq;
    return {m1.dot(m2), m2.dot(c1 * m2)};
  }
};

struct Info {
  double v1_mintime;
  int v2_stamps;
  double v3_cellsize;
  vector<int> v4_fpts, v4_frpts, v4_flpts, v4_brpts, v4_blpts;
  int v5_totalpts;
  int v6_validnm, v6_invalidnm;
  double v6_radius;
  vector<int> v7_cellpts;
  vector<int> v8_cellnms;
  vector<int> v9_cellgaus;
  int v10_totalvalidcells, v10_totalinvalidcells;

  string Brackets(const vector<int> vi) {
    if (!vi.size()) { return "[]"; }
    string ret("[");
    for (size_t i = 0; i < vi.size() - 1; ++i)
      ret += to_string(vi[i]) + ", ";
    ret += to_string(vi[vi.size() - 1]) + "]";
    return ret;
  }

  string ToString() {
    char c[1000];
    sprintf(c, "1. Minimum timestamp: %.6f s\n"
               "2. Augmented frames: 5 sensors x %d stamps = %d frames\n"
               "3. Cell size: %.2f m\n"
               "4. Points in each frame\n"
               "    - Front: %s\n"
               "    - Front Right: %s\n"
               "    - Front Left: %s\n"
               "    - Back Right: %s\n"
               "    - Back Left: %s\n"
               "5. Total points: %d\n"
               "6. Normals computed: %d valid (i.e., %d invalid)\n"
               "    - Radius of neighbors: %.2f\n"
               "7. Cells with [1, 2, ...] points:  %s\n"
               "8. Cells with [1, 2, ...] normals: %s\n"
               "9. Cells with valid gaussian: %s\n"
               "10. Total cells: %d valid (i.e., %d invalid)\n",
               v1_mintime, v2_stamps, v2_stamps * 5, v3_cellsize,
               Brackets(v4_fpts).c_str(), Brackets(v4_frpts).c_str(), Brackets(v4_flpts).c_str(),
               Brackets(v4_brpts).c_str(), Brackets(v4_blpts).c_str(), v5_totalpts,
               v6_validnm, v6_invalidnm, v6_radius, Brackets(v7_cellpts).c_str(),
               Brackets(v8_cellnms).c_str(), Brackets(v9_cellgaus).c_str(),
               v10_totalvalidcells, v10_totalinvalidcells);
    return string(c);
  }
};

pair<MatrixXd, Affine2d> ToPair(const common::PointCloudSensor &pcs) {
  MatrixXd fi(2, pcs.points.size());
  for (int i = 0; i < fi.cols(); ++i)
    fi.col(i) = Vector2d(pcs.points[i].x, pcs.points[i].y);
  Affine3d aff;
  tf2::fromMsg(pcs.origin, aff);
  Matrix3d mtx = Matrix3d::Identity();
  mtx.block<2, 2>(0, 0) = aff.matrix().block<2, 2>(0, 0);
  mtx.block<2, 1>(0, 2) = aff.matrix().block<2, 1>(0, 3);
  return make_pair(fi, Affine2d(mtx));
}

// start, ..., end, end+1
// <<------- T -------->>
vector<pair<MatrixXd, Affine2d>> Augment(
    const vector<common::EgoPointClouds> &vepcs, int start, int end,
    Affine2d &T, vector<Affine2d> &allT, Info &info) {
  vector<pair<MatrixXd, Affine2d>> ret;
  double dx = 0, dy = 0, dth = 0;
  for (int i = start; i <= end; ++i) {
    Affine2d T0i = Rotation2Dd(dth) * Translation2d(dx, dy);
    allT.push_back(T0i);
    for (const auto &pc : vepcs[i].pcs) {
      MatrixXd fi(2, pc.points.size());
      for (int i = 0; i < fi.cols(); ++i)
        fi.col(i) = Vector2d(pc.points[i].x, pc.points[i].y);
      Affine3d aff;
      tf2::fromMsg(pc.origin, aff);
      Matrix3d mtx = Matrix3d::Identity();
      mtx.block<2, 2>(0, 0) = aff.matrix().block<2, 2>(0, 0);
      mtx.block<2, 1>(0, 2) = aff.matrix().block<2, 1>(0, 3);
      ret.push_back(make_pair(fi, T0i * Affine2d(mtx)));
    }
    double dt = (vepcs[i + 1].stamp - vepcs[i].stamp).toSec();
    dx += vepcs[i].vxyt[0] * dt;
    dy += vepcs[i].vxyt[1] * dt;
    dth += vepcs[i].vxyt[2] * dt;
  }
  T = Rotation2Dd(dth) * Translation2d(dx, dy);
  /***** INFO ASSIGNS v1, v2, v4, v5 *****/
  info.v1_mintime = vepcs[start].stamp.toSec();
  info.v2_stamps = end - start + 1;
  info.v5_totalpts = 0;
  for (int i = start; i <= end; ++i) {
    for (int j = 0; j < 5; ++j) {
      if (vepcs[i].pcs[j].id == "front")
        info.v4_fpts.push_back(vepcs[i].pcs[j].points.size());
      else if (vepcs[i].pcs[j].id == "front right")
        info.v4_frpts.push_back(vepcs[i].pcs[j].points.size());
      else if (vepcs[i].pcs[j].id == "front left")
        info.v4_flpts.push_back(vepcs[i].pcs[j].points.size());
      else if (vepcs[i].pcs[j].id == "back right")
        info.v4_brpts.push_back(vepcs[i].pcs[j].points.size());
      else if (vepcs[i].pcs[j].id == "back left")
        info.v4_blpts.push_back(vepcs[i].pcs[j].points.size());
      info.v5_totalpts += vepcs[i].pcs[j].points.size();
    }
  }
  /***** END OF INFO ASSIGNS *****/
  return ret;
}

void InfoOfNDTMap(const NDTMap &map, double radius, Info &info) {
  int v6_validnm = 0, v6_invalidnm = 0;
  int n = (*max_element(map.begin(), map.end(), [](auto a, auto b) { return a->GetN() < b->GetN(); }))->GetN();
  vector<int> v7_cellpts(n), v8_cellnms(n), v9_cellgaus(n);
  for (auto cell : map) {
    ++v7_cellpts[cell->GetN() - 1];
    int valid = 0, invalid = 0; 
    for (auto nm : cell->GetNormals()) {
      if (nm.allFinite())
        ++valid;
      else
        ++invalid;
    }
    if (valid != 0)
      ++v8_cellnms[valid - 1];
    v6_validnm += valid;
    v6_invalidnm += invalid;
    if (cell->BothHasGaussian())
      ++v9_cellgaus[cell->GetN() - 1];
  }
  info.v3_cellsize = map.cell_size()(0);
  info.v6_radius = radius; 
  info.v6_validnm = v6_validnm;
  info.v6_invalidnm = v6_invalidnm;
  info.v7_cellpts = v7_cellpts;
  info.v8_cellnms = v8_cellnms;
  info.v9_cellgaus = v9_cellgaus;
  info.v10_totalvalidcells = accumulate(v9_cellgaus.begin(), v9_cellgaus.end(), 0);
  info.v10_totalinvalidcells = accumulate(v7_cellpts.begin(), v7_cellpts.end(), 0) - info.v10_totalvalidcells;
}

void cb(const std_msgs::Int32 &num) {
  int n = num.data;
  if (n < 0 || n >= (int)corres.size())
    return;
  pub7.publish(corres[n]);
  cout << "Mean: " << meancovs[n][0] << ", Cov: " << meancovs[n][1] << endl;
  geometry_msgs::Vector3 mc;
  mc.x = meancovs[n][0];
  mc.y = meancovs[n][1];
  mc.z = meancovs[n][2];
  pubmc.publish(mc);
}

string CorrespondenceMsg(NDTCell *p, NDTCell *q) {
  int vpN = 0, vqN = 0;
  for (const auto &n : p->GetNormals())
    if (n.allFinite())
      ++vpN;
  for (const auto &n : q->GetNormals())
    if (n.allFinite())
      ++vqN;
  char c[1000];
  auto pcen = p->GetCenter();
  auto pmean = p->GetPointMean();
  auto pcov = p->GetPointCov();
  auto npmean = p->GetNormalMean();
  auto npcov = p->GetNormalCov();
  auto qcen = q->GetCenter();
  auto qmean = q->GetPointMean();
  auto qcov = q->GetPointCov();
  auto nqmean = q->GetNormalMean();
  auto nqcov = q->GetNormalCov();
  sprintf(c, " p.N = %d (all %d)\n"
             " p.mean = (%.2f, %.2f)\n"
             " p.cov  = (%.2f, %.2f, %.2f, %.2f)\n"
             "np.mean = (%.2f, %.2f)\n"
             "np.cov  = (%.2f, %.2f, %.2f, %.2f\n"
             " q.N = %d (all %d)\n"
             " q.mean = (%.2f, %.2f)\n"
             " q.cov  = (%.2f, %.2f, %.2f, %.2f)\n"
             "nq.mean = (%.2f, %.2f)\n"
             "nq.cov  = (%.2f, %.2f, %.2f, %.2f)\n"
             "short = "
             "--p %.2f,%.2f,%.2f,%.2f,%.2f "
             "--np %.2f,%.2f,%.2f,%.2f,%.2f "
             "--cp %.2f,%.2f "
             "--q %.2f,%.2f,%.2f,%.2f,%.2f "
             "--nq %.2f,%.2f,%.2f,%.2f,%.2f "
             "--cq %.2f,%.2f", vpN, p->GetN(),
             pmean(0), pmean(1), pcov(0, 0), pcov(0, 1), pcov(1, 0), pcov(1, 1),
             npmean(0), npmean(1), npcov(0, 0), npcov(0, 1), npcov(1, 0), npcov(1, 1), vqN, q->GetN(),
             qmean(0), qmean(1), qcov(0, 0), qcov(0, 1), qcov(1, 0), qcov(1, 1),
             nqmean(0), nqmean(1), nqcov(0, 0), nqcov(0, 1), nqcov(1, 0), nqcov(1, 1),
             pmean(0), pmean(1), pcov(0, 0), pcov(0, 1), pcov(1, 1),
             npmean(0), npmean(1), npcov(0, 0), npcov(0, 1), npcov(1, 1),
             pcen(0), pcen(1),
             qmean(0), qmean(1), qcov(0, 0), qcov(0, 1), qcov(1, 1),
             nqmean(0), nqmean(1), nqcov(0, 0), nqcov(0, 1), nqcov(1, 1),
             qcen(0), qcen(1));
  return string(c);
}

int main(int argc, char **argv) {
  string path;
  int start_frame, frames;
  double cell_size, radius;
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help,h", "Produce help message")
      ("datapath,p", po::value<string>(&path)->required(), "Data Path")
      ("startframe,s", po::value<int>(&start_frame)->default_value(6), "Start Frame")
      ("frames,f", po::value<int>(&frames)->default_value(5), "Frames")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1), "Cell Size")
      ("radius,r", po::value<double>(&radius)->default_value(2.5), "Radius");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  ros::init(argc, argv, "test_0dis");
  ros::NodeHandle nh;

  vector<common::EgoPointClouds> vepcs;  
  common::SerializationInput(path, vepcs);
  cout << vepcs.size() << endl;

  vector<Affine2d> T611s, T1116s;
  Affine2d T611, T1116;
  Info infos, infot;
  auto datat = Augment(vepcs, start_frame, start_frame + frames - 1, T611, T611s, infot);
  auto mapt = MakeMap(datat, {0.0625, 0.0001}, {cell_size, radius});
  InfoOfNDTMap(mapt, radius, infot);

  auto datas = Augment(vepcs, start_frame + frames, start_frame + 2 * frames - 1, T1116, T1116s, infos);
  auto maps = MakeMap(datas, {0.0625, 0.0001}, {cell_size, radius});
  InfoOfNDTMap(maps, radius, infos);

  cout << infos.ToString() << endl;
  cout << infot.ToString() << endl;

  auto maps2 = maps.PseudoTransformCells(T611, true);

  auto kd = MakeKDTree(mapt);
  vector<MarkerArray> vps, vqs;
  vector<Vector2d> lines;
  int i = 0;
  for (auto cellp : maps2) {
    if (!cellp->BothHasGaussian()) { continue; }
    vector<int> idx(1);
    vector<float> dist2(1);
    pcl::PointXYZ pt(cellp->GetPointMean()(0), cellp->GetPointMean()(1), 0);
    int found = kd.nearestKSearch(pt, 1, idx, dist2);
    if (!found) { continue; }
    Vector2d npt((*kd.getInputCloud())[idx[0]].x, (*kd.getInputCloud())[idx[0]].y);
    auto cellq = mapt.GetCellForPoint(npt);
    if (!cellq || !cellq->BothHasGaussian()) { continue; }
    vps.push_back(MarkerArrayOfNDTCell(cellp.get()));
    vqs.push_back(MarkerArrayOfNDTCell2(cellq));
    lines.push_back(cellp->GetPointMean());
    lines.push_back(cellq->GetPointMean());
    auto linemarker = MarkerOfLines({cellp->GetPointMean(), cellq->GetPointMean()}, common::Color::kBlack, 1.0);
    corres.push_back(JoinMarkerArraysAndMarkers({vps.back(), vqs.back()}, {linemarker}));
    // meancovs.push_back(CostFunctions::OldMeanAndCov(cellp.get(), cellq));
    meancovs.push_back(CostFunctions::NewMeanAndCov(cellp.get(), cellq));
    meancovs.back().push_back((cellp->GetPointMean() - cellq->GetPointMean()).norm());
    cout << "Corres [" << i++ << "]:\n" << CorrespondenceMsg(cellp.get(), cellq) << endl;
    cout << "pn: " << endl;
    for (auto n : cellp->GetNormals())
      printf("(%.2f, %.2f)\n", n(0), n(1));
    cout << "qn: " << endl;
    for (auto n : cellq->GetNormals())
      printf("(%.2f, %.2f)\n", n(0), n(1));
  }

  cout << "Correspondences: " << corres.size() << endl;
  ros::Subscriber sub = nh.subscribe("idx", 0, cb);

  pubmc = nh.advertise<geometry_msgs::Vector3>("meancov", 0, true);
  ros::Publisher pub1 = nh.advertise<MarkerArray>("markers1", 0, true);  // source map after T
  ros::Publisher pub2 = nh.advertise<MarkerArray>("markers2", 0, true);  // target map
  ros::Publisher pub3 = nh.advertise<MarkerArray>("markers3", 0, true);  // source map after T (only corr)
  ros::Publisher pub4 = nh.advertise<MarkerArray>("markers4", 0, true);  // target map (only corr)
  ros::Publisher pub5 = nh.advertise<MarkerArray>("markers5", 0, true);  // source sensor
  ros::Publisher pub6 = nh.advertise<MarkerArray>("markers6", 0, true);  // target sensor
  pub7 = nh.advertise<MarkerArray>("markers7", 0, true);  // correspondences
  ros::Publisher pub8 = nh.advertise<MarkerArray>("markers8", 0, true);  // source ego car
  ros::Publisher pub9 = nh.advertise<MarkerArray>("markers9", 0, true);  // target ego car

  ros::Publisher pb1 = nh.advertise<Marker>("marker1", 0, true);  // correspondences
  ros::Publisher pb2 = nh.advertise<Marker>("marker2", 0, true);  // source points after T
  ros::Publisher pb3 = nh.advertise<Marker>("marker3", 0, true);  // target points

  pub1.publish(MarkerArrayOfNDTMap(maps2));
  pub2.publish(MarkerArrayOfNDTMap(mapt, true));
  pub3.publish(JoinMarkerArraysAndMarkers(vps));
  pub4.publish(JoinMarkerArraysAndMarkers(vqs));
  vector<Affine2d> affs, afft;
  transform(datas.begin(), datas.end(), back_inserter(affs), [&T611](auto a) { return T611 * a.second; });
  transform(datat.begin(), datat.end(), back_inserter(afft), [](auto a) { return a.second; });
  pub5.publish(MarkerArrayOfSensor(affs));
  pub6.publish(MarkerArrayOfSensor(afft));

  pb1.publish(MarkerOfLines(lines, common::Color::kBlack, 1.0));
  pb2.publish(MarkerOfPoints(PointsOfNDTMap(maps2), 0.1));
  pb3.publish(MarkerOfPoints(PointsOfNDTMap(mapt), 0.1, common::Color::kRed));
  vector<Vector2d> cars, cart;
  transform(T1116s.begin(), T1116s.end(), back_inserter(cars), [&T611](auto a) { return T611 * a.translation(); });
  transform(T611s.begin(), T611s.end(), back_inserter(cart), [](auto a) { return a.translation(); });
  auto mcarsp = MarkerOfPoints(cars, 0.1, common::Color::kBlack);
  auto mcartp = MarkerOfPoints(cart, 0.1, common::Color::kBlack);
  auto mcars = MarkerOfLinesByEndPoints(cars, common::Color::kRed, 1.0);
  auto mcart = MarkerOfLinesByEndPoints(cart, common::Color::kLime, 1.0);
  pub8.publish(JoinMarkers({mcars, mcarsp}));
  pub9.publish(JoinMarkers({mcart, mcartp}));
  ros::spin();
}
