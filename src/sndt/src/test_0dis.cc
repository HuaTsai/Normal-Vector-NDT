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
#include <sndt/wrapper.hpp>
#include <boost/program_options.hpp>
#include <common/EgoPointClouds.h>
#include <sndt/ndt_conversions.hpp>

using namespace std;
namespace po = boost::program_options;

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
    Affine2d &T, Info &info) {
  vector<pair<MatrixXd, Affine2d>> ret;
  T = Affine2d::Identity();
  double dx = 0, dy = 0, dth = 0;
  for (int i = start; i <= end; ++i) {
    double dt = (vepcs[i + 1].stamp - vepcs[i].stamp).toSec();
    dx += vepcs[i].vxyt[0] * dt;
    dy += vepcs[i].vxyt[1] * dt;
    dth += vepcs[i].vxyt[2] * dt;
    Affine2d T0i = Rotation2Dd(dth) * Translation2d(dx, dy);
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
  }
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

  Affine2d T611, T1116;
  Info infos, infot;
  auto datat = Augment(vepcs, start_frame, start_frame + frames - 1, T611, infot);
  auto mapt = MakeMap(datat, {0.0625, 0.0001}, {cell_size, radius});
  cout << mapt.ToString() << endl;
  InfoOfNDTMap(mapt, radius, infot);

  auto datas = Augment(vepcs, start_frame + frames, start_frame + 2 * frames - 1, T1116, infos);
  auto maps = MakeMap(datas, {0.0625, 0.0001}, {cell_size, radius});
  cout << maps.ToString() << endl;
  InfoOfNDTMap(maps, radius, infos);

  cout << infos.ToString() << endl;
  cout << infot.ToString() << endl;

  auto kd = MakeKDTree(mapt);
  int i = 0;
  vector<visualization_msgs::MarkerArray> vps, vqs;
  vector<Vector2d> lines;
  auto maps2 = maps.PseudoTransformCells(T611);
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
    vqs.push_back(MarkerArrayOfNDTCell(cellq));
    lines.push_back(cellp->GetPointMean());
    lines.push_back(cellq->GetPointMean());
  }

  ros::Publisher pub = nh.advertise<visualization_msgs::MarkerArray>("marker", 0, true);
  ros::Publisher pub2 = nh.advertise<visualization_msgs::MarkerArray>("marker2", 0, true);
  ros::Publisher pub3 = nh.advertise<visualization_msgs::MarkerArray>("marker3", 0, true);
  ros::Publisher pub4 = nh.advertise<visualization_msgs::MarkerArray>("marker4", 0, true);
  ros::Publisher pub5 = nh.advertise<visualization_msgs::Marker>("marker5", 0, true);
  ros::Publisher pub6 = nh.advertise<visualization_msgs::MarkerArray>("marker6", 0, true);
  ros::Publisher pub7 = nh.advertise<visualization_msgs::MarkerArray>("marker7", 0, true);
  pub.publish(MarkerArrayOfNDTMap(maps2));
  pub2.publish(MarkerArrayOfNDTMap(mapt));
  pub3.publish(JoinMarkerArraysAndMarkers(vps));
  pub4.publish(JoinMarkerArraysAndMarkers(vqs));
  pub5.publish(MarkerOfLines(lines, common::Color::kRed, 1.0));
  vector<Affine2d> afft, affs;
  transform(datat.begin(), datat.end(), back_inserter(afft), [](auto a) { return a.second; });
  transform(datas.begin(), datas.end(), back_inserter(affs), [](auto a) { return a.second; });
  pub6.publish(MarkerArrayOfSensor(afft));
  pub7.publish(MarkerArrayOfSensor(affs));
  ros::spin();
}
