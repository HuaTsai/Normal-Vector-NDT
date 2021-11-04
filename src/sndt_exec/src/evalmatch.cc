#include <pcl/filters/voxel_grid.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sndt/cost_functors.h>
#include <sndt/matcher.h>
#include <sndt/visuals.h>
#include <std_msgs/Int32.h>
#include <tqdm/tqdm.h>

#include <boost/program_options.hpp>
#include <sndt_exec/wrapper.h>

using namespace std;
using namespace Eigen;
using namespace visualization_msgs;
namespace po = boost::program_options;

vector<vector<visualization_msgs::MarkerArray>> ms;
vector<vector<string>> ss;
ros::Publisher pub1, pub2, pub3, pub4, pub5, pub6, pub7, pub8;
vector<Affine2d> Ts;
vector<Affine2d> Txs;
vector<vector<Vector2d>> us, uns;
vector<vector<Matrix2d>> cs, cns;

MarkerArray Empty() {
  MarkerArray empty;
  for (int i = 0; i < 30; ++i) {
    Marker m;
    m.id = i;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.type = Marker::DELETEALL;
    empty.markers.push_back(m);
  }
  return empty;
}

void cb(const std_msgs::Int32 &num) {
  int n = num.data;
  pub1.publish(Empty());
  pub2.publish(Empty());
  pub3.publish(Empty());
  pub1.publish(ms[n][0]);
  pub2.publish(ms[n][1]);
  pub3.publish(ms[n][2]);
  // cout << "tgt:" << endl << ss[n][0] << endl;
  // cout << "src:" << endl << ss[n][1] << endl;
  // cout << "src2:" << endl << ss[n][2] << endl;
  auto up = us[n][1], unp = uns[n][1];
  auto cp = cs[n][1], cnp = cns[n][1];
  auto uq = us[n][0], unq = uns[n][0];
  auto cq = cs[n][0], cnq = cns[n][0];
  auto aff = Translation2d(13.7548, -5.98377) * Rotation2Dd(0);
  double cost = SNDTCostFunctor2::Cost(up, cp, unp, cnp, uq, cq, unq, cnq);
  double cost2 =
      SNDTCostFunctor2::Cost(up, cp, unp, cnp, uq, cq, unq, cnq, aff);
  double cost3 = SNDTCostFunctor3::Cost(up, cp, unp, uq, cq, unq);
  double cost4 = SNDTCostFunctor3::Cost(up, cp, unp, uq, cq, unq, aff);
  double cost5 = D2DNDTCostFunctor2::Cost(up, cp, uq, cq);
  double cost6 = D2DNDTCostFunctor2::Cost(up, cp, uq, cq, aff);
  // printf("up = np.array([[%f], [%f]])\n", up(0), up(1));
  // printf("cp = np.array([[%f, %f], [%f, %f]])\n", cp(0, 0), cp(0, 1), cp(1,
  // 0), cp(1, 1)); printf("unp = np.array([[%f], [%f]])\n", unp(0), unp(1));
  // printf("cnp = np.array([[%f, %f], [%f, %f]])\n", cnp(0, 0), cnp(0, 1),
  // cnp(1, 0), cnp(1, 1)); printf("uq = np.array([[%f], [%f]])\n", uq(0),
  // uq(1)); printf("cq = np.array([[%f, %f], [%f, %f]])\n", cq(0, 0), cq(0, 1),
  // cq(1, 0), cq(1, 1)); printf("unq = np.array([[%f], [%f]])\n", unq(0),
  // unq(1)); printf("cnq = np.array([[%f, %f], [%f, %f]])\n", cnq(0, 0), cnq(0,
  // 1), cnq(1, 0), cnq(1, 1)); cout << "SNDT      => " << cost << ", " << 0.5 *
  // cost * cost << endl; cout << "SNDT(gt)  => " << cost2 << ", " << 0.5 *
  // cost2 * cost2 << endl; cout << "SNDT2     => " << cost3 << ", " << 0.5 *
  // cost3 * cost3 << endl; cout << "SNDT2(gt) => " << cost4 << ", " << 0.5 *
  // cost4 * cost4 << endl; cout << "D2D       => " << cost5 << ", " << 0.5 *
  // cost5 * cost5 << endl; cout << "D2D(gt)   => " << cost6 << ", " << 0.5 *
  // cost6 * cost6 << endl;
}

int main(int argc, char **argv) {
  Affine3d aff3 = Translation3d(0.943713, 0.000000, 1.840230) *
                  Quaterniond(0.707796, -0.006492, 0.010646, -0.706307);
  aff3 = Conserve2DFromAffine3d(aff3);
  Affine2d aff2 = Translation2d(aff3.translation()(0), aff3.translation()(1)) *
                  Rotation2Dd(aff3.rotation().block<2, 2>(0, 0));

  int n, m;
  double cell_size, huber, voxel, r, radius;
  string data;
  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", "Produce help message")
      ("data,d", po::value<string>(&data)->required(), "Data (logxx)")
      ("cellsize,c", po::value<double>(&cell_size)->default_value(1.5), "Cell Size")
      ("voxel,v", po::value<double>(&voxel)->default_value(0), "Downsample voxel")
      ("huber,u", po::value<double>(&huber)->default_value(1.0), "Use Huber loss")
      ("radius,a", po::value<double>(&radius)->default_value(1.5), "Search radius")
      ("n,n", po::value<int>(&n)->default_value(0), "n")
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
  ros::Subscriber sub1 = nh.subscribe("idx", 0, cb);

  MarkerArray empty;
  for (int i = 0; i < 30; ++i) {
    Marker m;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.type = Marker::DELETE;
    empty.markers.push_back(m);
  }

  vector<sensor_msgs::PointCloud2> vpc;
  SerializationInput(JoinPath(GetDataPath(data), "lidar.ser"), vpc);

  auto tgt = PCMsgTo2D(vpc[n], voxel);
  transform(tgt.begin(), tgt.end(), tgt.begin(),
            [&aff2](auto p) { return aff2 * p; });

  int samples = 1;
  auto affs = RandomTransformGenerator2D(r).Generate(samples);
  cout << "r = " << r << endl;

  for (auto aff : affs) {
    aff = Translation2d(-13.7548, 5.98377) * Rotation2Dd(0);
    cout << "Offset: " << XYTDegreeFromAffine2d(aff).transpose() << endl;
    std::vector<Eigen::Vector2d> src(tgt.size());
    transform(tgt.begin(), tgt.end(), src.begin(),
              [&aff](auto p) { return aff * p; });
    vector<pair<vector<Vector2d>, Affine2d>> datat{
        {tgt, Eigen::Affine2d::Identity()}};
    vector<pair<vector<Vector2d>, Affine2d>> datas{
        {src, Eigen::Affine2d::Identity()}};

    SNDTParameters params6;
    params6.cell_size = cell_size;
    params6.radius = radius;
    params6.r_variance = params6.t_variance = 0;
    auto tgt6 = MakeSNDTMap(datat, params6);
    auto src6 = MakeSNDTMap(datas, params6);
    // auto T = SNDTMatch(tgt6, src6, params6);
    // pub3.publish(MarkerArrayOfSNDTMap(tgt6, true));
    // cout << "Result: " << XYTDegreeFromAffine2d(aff * T).transpose() << endl;

    // auto corr = params6._corres[0];
    // cout << "Correspondences: " << corr.size() << endl;
    // for (size_t i = 0; i < corr.size(); ++i) {
    // auto tcell = tgt6[corr[i].second];
    // auto scell = src6[corr[i].first];
    auto tcell = tgt6[20];
    auto scell = src6[20];
    auto tpts = tcell->GetPoints();
    auto spts = scell->GetPoints();

    SNDTParameters params;
    params.cell_size = cell_size;
    params.radius = radius;
    params.r_variance = params.t_variance = 0;
    auto Tx =
        SNDTCellMatch(tcell, scell, params, Eigen::Affine2d::Identity(), m);
    for (size_t i = 0; i < params._costs.size(); ++i) {
      cout << params._costs[i][0].first << " ("
           << XYTDegreeFromAffine2d(params._sols[i].front()).transpose() << ")"
           << ((i + 1 != params._costs.size()) ? " -> " : "");
    }
    cout << " => (" << XYTDegreeFromAffine2d(Tx).transpose() << ")"
         << "\n";
    vector<double> xs, ys, ts;
    for (auto sols : params._sols) {
      for (auto sol : sols) {
        auto xyt = XYTDegreeFromAffine2d(sol);
        xs.push_back(xyt(0));
        ys.push_back(xyt(1));
        ts.push_back(xyt(2));
      }
    }
    // cout << "x = ["; for (auto x : xs) cout << x << ", "; cout << "]" <<
    // endl; cout << "y = ["; for (auto y : ys) cout << y << ", "; cout << "]"
    // << endl; cout << "t = ["; for (auto t : ts) cout << t << ", "; cout <<
    // "]" << endl;

    auto c = src6.PseudoTransformCells(Tx, true)[20];
    ms.push_back({MarkerArrayOfSNDTCell2(tcell), MarkerArrayOfSNDTCell(scell),
                  MarkerArrayOfSNDTCell(c.get())});
    ss.push_back({tcell->ToString(), scell->ToString(), c->ToString()});
    us.push_back(
        {tcell->GetPointMean(), scell->GetPointMean(), c->GetPointMean()});
    uns.push_back(
        {tcell->GetNormalMean(), scell->GetNormalMean(), c->GetNormalMean()});
    cs.push_back(
        {tcell->GetPointCov(), scell->GetPointCov(), c->GetPointCov()});
    cns.push_back(
        {tcell->GetNormalCov(), scell->GetNormalCov(), c->GetNormalCov()});
    std_msgs::Int32 num;
    num.data = 0;
    cb(num);
    ros::spin();
    // }
    cout << "Ready!" << endl;
  }
}
