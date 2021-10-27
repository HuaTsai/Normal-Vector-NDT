// Same Scans with Given Transformation Range
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <std_msgs/Int32.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.hpp>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;

template <typename T>
double Avg(const T &c) {
  return accumulate(c.begin(), c.end(), 0.) / c.size();
}

void Q1MedianQ3(vector<double> &data) {
  sort(data.begin(), data.end());
  printf("[%g, %g, %g, %g, %g]\n", data[0], data[data.size() / 4],
         data[data.size() / 2], data[data.size() / 4 * 3],
         data[data.size() - 1]);
  // cout << "Min: " << data[0] << endl;
  // cout << " Q1: " << data[data.size() / 4] << endl;
  // cout << "Mid: " << data[data.size() / 2] << endl;
  // cout << " Q3: " << data[data.size() / 4 * 3] << endl;
  // cout << "Max: " << data[data.size() - 1] << endl;
}

vector<Vector2d> PCMsgTo2D(const sensor_msgs::PointCloud2 &msg, double voxel) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(msg, *pc);
  if (voxel != 0) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(pc);
    vg.setLeafSize(voxel, voxel, voxel);
    vg.filter(*pc);
  }

  vector<Vector2d> ret;
  for (const auto &pt : *pc)
    if (isfinite(pt.x) && isfinite(pt.y) && isfinite(pt.z))
      ret.push_back(Vector2d(pt.x, pt.y));
  return ret;
}

double RMS(const vector<Vector2d> &tgt,
           const vector<Vector2d> &src,
           const Affine2d &aff = Affine2d::Identity()) {
  vector<Vector2d> src2(src.size());
  transform(src.begin(), src.end(), src2.begin(),
            [&aff](auto p) { return aff * p; });
  double ret = 0;
  for (size_t i = 0; i < src2.size(); ++i)
    ret += (src2[i] - tgt[i]).squaredNorm();
  return sqrt(ret / src2.size());
}

void ShowRMS(const CommonParameters &params,
             const vector<Vector2d> &tgt,
             const vector<Vector2d> &src) {
  cout << "[";
  for (auto sols : params._sols) {
    for (auto T : sols) cout << RMS(tgt, src, T) << ", ";
  }
  cout << " ]";
  cout << endl;
}

void ShowAllSols(const CommonParameters &params) {
  for (size_t i = 0; i < params._sols.size(); ++i) {
    cout << i << ": " << params._sols[i].front().translation().transpose()
         << " -> " << params._sols[i].back().translation().transpose() << endl;
  }
}

vector<vector<visualization_msgs::MarkerArray>> ms;
ros::Publisher pub1, pub2, pub3, pub4, pub5, pub6, pub7, pub8;
vector<Affine2d> Ts;
vector<Affine2d> Txs;

void cb(const std_msgs::Int32 &idx) {
  int n = idx.data;
  pub1.publish(ms[n][0]);
  pub2.publish(ms[n][1]);
  pub3.publish(ms[n][2]);
  pub4.publish(ms[n][3]);
  cout << Ts[n].translation().transpose() << endl;
  cout << Txs[n].translation().transpose() << endl;
}

void OutlierRemove(vector<Vector2d> &data) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto pt : data) pc->push_back(pcl::PointXYZ(pt(0), pt(1), 0));
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> filter(true);
  filter.setInputCloud(pc);
  filter.setMeanK(8);
  filter.setStddevMulThresh(1.0);
  vector<int> indices;
  filter.filter(indices);
  data.clear();
  for (auto i : indices) data.push_back(Vector2d(pc->at(i).x, pc->at(i).y));
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n, m;
  double cell_size, huber, voxel, x, y, t, r, radius;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("huber,u", po::value<double>(&huber)->default_value(1.0), "Use Huber loss")
      ("radius,a", po::value<double>(&radius)->default_value(1.5), "Use Huber loss")
      ("n,n", po::value<int>(&n)->default_value(0), "n")
      ("x,x", po::value<double>(&x)->default_value(0), "x")
      ("y,y", po::value<double>(&y)->default_value(0), "y")
      ("t,t", po::value<double>(&t)->default_value(0), "t")
      ("r,r", po::value<double>(&r)->default_value(15), "r")
      ("m,m", po::value<int>(&m)->default_value(1), "r");
  // clang-format on
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }
  po::notify(vm);

  ros::init(argc, argv, "evalsndt");
  ros::NodeHandle nh;
  pub1 = nh.advertise<MarkerArray>("markers1", 0, true);
  pub2 = nh.advertise<MarkerArray>("markers2", 0, true);
  pub3 = nh.advertise<MarkerArray>("markers3", 0, true);
  pub4 = nh.advertise<MarkerArray>("markers4", 0, true);
  // pub5 = nh.advertise<MarkerArray>("markers5", 0, true);
  // pub6 = nh.advertise<MarkerArray>("markers6", 0, true);
  // pub7 = nh.advertise<MarkerArray>("markers7", 0, true);
  // pub8 = nh.advertise<MarkerArray>("markers8", 0, true);

  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);

  auto tgt = PCMsgTo2D(vpc[n], voxel);
  transform(tgt.begin(), tgt.end(), tgt.begin(),
            [&aff2](auto p) { return aff2 * p; });
  // OutlierRemove(tgt);

#if 1
  // ros::Subscriber sub = nh.subscribe("idx", 0, cb);
  int samples = 15;
  auto affs = RandomTransformGenerator2D(r).Generate(samples);
  cout << "r = " << r << endl;
  // vector<double> rms6;
  for (auto aff : affs) {
    std::vector<Eigen::Vector2d> src(tgt.size());
    transform(tgt.begin(), tgt.end(), src.begin(),
              [&aff](auto p) { return aff * p; });
    vector<pair<vector<Vector2d>, Affine2d>> datat{
        {tgt, Eigen::Affine2d::Identity()}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{
        {src, Eigen::Affine2d::Identity()}};

    CommonParameters *params;
    Affine2d T;
    ICPParameters params1;
    Pt2plICPParameters params2;
    SICPParameters params3;
    P2DNDTParameters params4;
    D2DNDTParameters params5;
    SNDTParameters params6;
    D2DNDTParameters params7;
    if (m == 1) {
      params = &params1;
      auto tgt1 = MakePoints(datat, params1);
      auto src1 = MakePoints(datas, params1);
      T = ICPMatch(tgt1, src1, params1);
    } else if (m == 2) {
      params = &params2;
      params2.radius = radius;
      auto tgt2 = MakePoints(datat, params2);
      auto src2 = MakePoints(datas, params2);
      T = Pt2plICPMatch(tgt2, src2, params2);
    } else if (m == 3) {
      params = &params3;
      params3.radius = radius;
      auto tgt3 = MakePoints(datat, params3);
      auto src3 = MakePoints(datas, params3);
      T = SICPMatch(tgt3, src3, params3);
    } else if (m == 4) {
      params = &params4;
      params4.cell_size = cell_size;
      params4.r_variance = params4.t_variance = 0;
      auto tgt4 = MakeNDTMap(datat, params4);
      auto src4 = MakePoints(datas, params4);
      T = P2DNDTMatch(tgt4, src4, params4);
    } else if (m == 5) {
      params = &params5;
      params5.cell_size = cell_size;
      params5.r_variance = params5.t_variance = 0;
      auto tgt5 = MakeNDTMap(datat, params5);
      auto src5 = MakeNDTMap(datas, params5);
      T = D2DNDTMatch(tgt5, src5, params5);
    } else if (m == 6) {
      params = &params6;
      params6.cell_size = cell_size;
      params6.radius = radius;
      params6.r_variance = params6.t_variance = 0;
      auto tgt6 = MakeSNDTMap(datat, params6);
      auto src6 = MakeSNDTMap(datas, params6);
      T = SNDTMatch(tgt6, src6, params6);
      pub3.publish(MarkerArrayOfSNDTMap(tgt6, true));
    } else if (m == 7) {
      params = &params7;
      params7.cell_size = cell_size;
      params7.r_variance = params7.t_variance = 0;
      auto tgt7 = MakeNDTMap(datat, params7);
      auto src7 = MakeNDTMap(datas, params7);
      T = SNDTMatch2(tgt7, src7, params7);
    }

    cout << (((aff * T).translation().isZero(1)) ? "success" : "fail") << ", ";
    cout << "Iterations: " << params->_iteration << " & "
         << params->_ceres_iteration << endl;
    for (auto tf : params->_sols) {
      vector<Vector2d> src2(src.size());
      transform(src.begin(), src.end(), src2.begin(),
                [&tf](auto p) { return tf.front() * p; });
      pub1.publish(JoinMarkers({MarkerOfPoints(tgt, 0.5, Color::kRed)}));
      pub2.publish(JoinMarkers({MarkerOfPoints(src2)}));
      ros::Rate(10).sleep();
    }
    // for (size_t i = 0; i < params->_costs.size(); ++i) {
    //   cout << "Iter " << i << " = ";
    //   Q1MedianQ3(params->_costs[i]);
    // }

    // rms6.push_back(RMS(tgt, src, params._sols[0].back()));
    // ms.push_back(vector<MarkerArray>(4));
    // ms.back()[0] = JoinMarkers({MarkerOfPoints(tgt, 0.5, Color::kRed)});
    // ms.back()[1] = JoinMarkers({MarkerOfPoints(src)});
    // vector<Eigen::Vector2d> srcTx(src.size());
    // transform(src.begin(), src.end(), srcTx.begin(), [&Tx](auto p) { return
    // Tx * p; }); ms.back()[2] = JoinMarkers({MarkerOfPoints(srcTx)});
    // vector<Eigen::Vector2d> srcT(src.size());
    // transform(src.begin(), src.end(), srcT.begin(), [&T](auto p) { return T *
    // p; }); ms.back()[3] = JoinMarkers({MarkerOfPoints(srcT)});
  }
  cout << "Ready!" << endl;
#endif

#if 0
  vector<Eigen::Vector2d> src(tgt.size());
  transform(tgt.begin(), tgt.end(), src.begin(), [&x, &y, &t](auto p) { return Translation2d(x, y) * p; });
  vector<pair<vector<Vector2d>, Affine2d>> datat{{tgt, Eigen::Affine2d::Identity()}};
  vector<pair<vector<Vector2d>, Affine2d>> datas{{src, Eigen::Affine2d::Identity()}};

  // SICP
  // SICPParameters params;
  // auto tgt6 = MakePoints(datat, params);
  // auto src6 = MakePoints(datas, params);
  // auto T = SICPMatch(tgt6, src6, params);
  // auto Tx = params._sols[0].back();
  // for (auto cost : params._costs) {
  //   cout << Avg(cost) << ", ";
  // }
  // cout << endl;

  // NDT
  // NDTParameters params;
  // params.r_variance = params.t_variance = 0;
  // auto tgt6 = MakeNDTMap(datat, params);
  // auto src6 = MakeNDTMap(datas, params);
  // auto T = D2DNDTMatch(tgt6, src6, params);
  // auto Tx = params._sols[0].back();
  // for (auto cost : params._costs) {
  //   cout << Avg(cost) << ", ";
  // }
  // cout << endl;

  // PNDT
  // NDTParameters params;
  // params.r_variance = params.t_variance = 0;
  // params.cell_size = cell_size, params.huber = huber;
  // auto tgt6 = MakeNDTMap(datat, params);
  // auto src6 = MakePoints(datas, params);
  // auto T = P2DNDTMatch(tgt6, src6, params);
  // auto Tx = params._sols[0].back();
  // for (auto cost : params._costs) {
  //   cout << Avg(cost) << ", ";
  // }
  // cout << endl;

  // SNDT
  SNDTParameters params;
  params.r_variance = params.t_variance = 0;
  params.cell_size = cell_size, params.huber = huber;
  auto tgt6 = MakeSNDTMap(datat, params);
  auto src6 = MakeSNDTMap(datas, params);
  auto T = SNDTMatch(tgt6, src6, params);
  auto Tx = params._sols[0].back();
  cout << Avg(params._costs[0]) << endl;
  Q1MedianQ3(params._costs[0]);

  ShowAllSols(params);
  pub1.publish(JoinMarkers({MarkerOfPoints(tgt, 0.5, Color::kRed)}));
  pub2.publish(JoinMarkers({MarkerOfPoints(src)}));

  vector<Eigen::Vector2d> srcTx(src.size());
  transform(src.begin(), src.end(), srcTx.begin(), [&Tx](auto p) { return Tx * p; });
  pub3.publish(JoinMarkers({MarkerOfPoints(srcTx)}));

  vector<Eigen::Vector2d> srcT(src.size());
  transform(src.begin(), src.end(), srcT.begin(), [&T](auto p) { return T * p; });
  pub4.publish(JoinMarkers({MarkerOfPoints(srcT)}));

  // SNDT
  // pub5.publish(JoinMarkerArrays({MarkerArrayOfSNDTMap(tgt6, true)}));
  // pub6.publish(JoinMarkerArrays({MarkerArrayOfSNDTMap(src6)}));
  // pub7.publish(JoinMarkerArrays({MarkerArrayOfSNDTMap(src6.PseudoTransformCells(T, true))}));
  // pub8.publish(JoinMarkerArrays({MarkerArrayOfSNDTMap(src6.PseudoTransformCells(Tx, true))}));

  ros::spin();
#endif
}