#include <bits/stdc++.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/features/normal_3d.h>
#include "sndt/ndt_map_2d.hpp"
#include "sndt/ndt_conversions.hpp"
#include "sndt/ndt_matcher_2d.hpp"
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>
#include <tf2_ros/static_transform_broadcaster.h>

using namespace std;

pcl::PointCloud<pcl::PointXY> GenerateData(const Eigen::Matrix3d &T = Eigen::Matrix3d::Identity()) {
  pcl::PointCloud<pcl::PointXY> ret;
  for (int i = 0; i < 30; ++i) {
    pcl::PointXY pt;
    pt.x = i;
    pt.y = 0.2 * i * sin(0.5 * i);
    pt.x = T(0, 0) * pt.x + T(0, 1) * pt.y + T(0, 2);
    pt.y = T(1, 0) * pt.x + T(1, 1) * pt.y + T(1, 2);
    ret.points.push_back(pt);
  }
  return ret;
}

vector<Eigen::Vector2d> ComputeNormals(const pcl::PointCloud<pcl::PointXY> &pc,
                                       const Eigen::Matrix3d &T = Eigen::Matrix3d::Identity()) {
  vector<Eigen::Vector2d> ret;
  Eigen::Vector3d p0(-1, 0.2 * (-1) * sin(0.5 * (-1)), 1);
  Eigen::Vector3d p29(30, 0.2 * 30 * sin(0.5 * 30), 1);
  p0 = T * p0;
  p29 = T * p29;
  for (int i = 0; i < 30; ++i) {
    Eigen::Vector2d normal;
    double dx, dy;
    if (i == 0) {
      dx = pc.points.at(i + 1).x - p0(0);
      dy = pc.points.at(i + 1).y - p0(1);
    } else if (i == 29) {
      dx = p29(0) - pc.points.at(i - 1).x;
      dy = p29(1) - pc.points.at(i - 1).y;
    } else {
      dx = pc.points.at(i + 1).x - pc.points.at(i - 1).x;
      dy = pc.points.at(i + 1).y - pc.points.at(i - 1).y;
    }
    normal << -dy, dx;
    normal.normalize();
    ret.push_back(normal);
  }
  return ret;
}

pcl::PointCloud<pcl::PointXY> ComputeNormalsPCL(const pcl::PointCloud<pcl::PointXY> &pc,
                                                const Eigen::Matrix3d &T = Eigen::Matrix3d::Identity()) {
  pcl::PointCloud<pcl::PointXY> ret;
  auto normals = ComputeNormals(pc, T);
  for (const auto &normal : normals) {
    pcl::PointXY n;
    n.x = normal(0);
    n.y = normal(1);
    ret.points.push_back(n);
  }
  return ret;
}

visualization_msgs::MarkerArray NormalsMarkerArray(
    const pcl::PointCloud<pcl::PointXY> &pc,
    const pcl::PointCloud<pcl::PointXY> &normals) {
  visualization_msgs::MarkerArray ret;
  for (size_t i = 0; i < pc.points.size(); ++i) {
    Eigen::Vector3d start(pc.points.at(i).x, pc.points.at(i).y, 0);
    Eigen::Vector3d end = start + Eigen::Vector3d(normals.at(i).x, normals.at(i).y, 0);
    ret.markers.push_back(common::MakeArrowMarkerByEnds(i, "map", start, end, common::Color::kLime));
  }
  return ret;
}

Eigen::Matrix3d GenTF(double x, double y, double th) {
  Eigen::Matrix3d ret;
  double rad = th * M_PI / 180.;
  ret << cos(rad), -sin(rad), x, sin(rad), cos(rad), y, 0, 0, 1;
  return ret;
}

pcl::PointCloud<pcl::PointXYZ> PC3D(const pcl::PointCloud<pcl::PointXY> &pc,
                                    Eigen::Matrix3d T = Eigen::Matrix3d::Identity()) {
  pcl::PointCloud<pcl::PointXYZ> pc3d;
  for (const auto &elem : pc.points) {
    Eigen::Vector3d pt = T * Eigen::Vector3d(elem.x, elem.y, 1);
    pcl::PointXYZ p;
    p.x = pt(0);
    p.y = pt(1);
    pc3d.points.push_back(p);
  }
  pc3d.header.frame_id = "map";
  return pc3d;
}

double diff(Eigen::Matrix3d T1, Eigen::Matrix3d T2) {
  Eigen::Vector3d Tdiff(T1(0, 2) - T2(0, 2), T1(1, 2) - T2(1, 2), acos(T1(0, 0)) - acos(T2(0, 0)));
  return Tdiff.norm();
}

class RandomTransformGenerator2D {
 public:
  RandomTransformGenerator2D() {
    dre = new default_random_engine();
  }
  ~RandomTransformGenerator2D() {
    delete dre;
  }
  void SetTranslationRadiusBound(double min, double max) {
    radius_min = min;
    radius_max = max;
  }
  void SetRotationDegreeBound(double min, double max) {
    angle_min = min * M_PI / 180.;
    angle_max = max * M_PI / 180.;
  }
  vector<Eigen::Matrix3d> Generate(int sizes) {
    vector<Eigen::Matrix3d> ret(sizes, Eigen::Matrix3d::Identity());
    uniform_real_distribution<> radius_urd(-radius_max, radius_max);
    int n = 0;
    while (n < sizes) {
      Eigen::Vector2d vec(radius_urd(*dre), radius_urd(*dre));
      if (vec.norm() < radius_max && vec.norm() > radius_min) {
        ret.at(n).block<2, 1>(0, 2) = vec;
        ++n;
      }
    }
    uniform_real_distribution<> angle_urd(angle_min, angle_max);
    for (int i = 0; i < sizes; ++i) {
      double angle = angle_urd(*dre);
      ret.at(i)(0, 0) = cos(angle);
      ret.at(i)(0, 1) = -sin(angle);
      ret.at(i)(1, 0) = sin(angle);
      ret.at(i)(1, 1) = cos(angle);
    }
    return ret;
  }
 private:
  default_random_engine *dre;
  int mode;
  double radius_min;
  double radius_max;
  double angle_min;
  double angle_max;
};

int main(int argc, char **argv) {

#if 0
  pcl::PointCloud<pcl::PointXY> pc = GenerateData();
  auto normals = ComputeNormalsPCL(pc);
  NDTMap map(new LazyGrid2D(2.5));
  map.loadPointCloud(pc, normals);
  map.computeNDTCells();

  int samples = 1000;
  RandomTransformGenerator2D rtg;
  // vector<double> rbound = {0, 1, 2, 3};//, 4, 5, 6};
  // vector<double> abound = {0, 10, 20, 30};//, 40, 50, 60};
  vector<double> rbound = {0, 1, 2, 3, 4, 5, 6};
  vector<double> abound = {0, 10, 20, 30, 40, 50, 60};
  for (size_t rmin = 0; rmin < rbound.size() - 1; ++rmin) {
    rtg.SetTranslationRadiusBound(rbound.at(rmin), rbound.at(rmin + 1));
    for (size_t amin = 0; amin < abound.size() - 1; ++amin) {
      rtg.SetRotationDegreeBound(abound.at(amin), abound.at(amin + 1));
      printf("(%.2f, %.2f) x (%.2f, %.2f): ",
             rbound.at(rmin), rbound.at(rmin + 1), abound.at(amin),
             abound.at(amin + 1));
      auto mtxs = rtg.Generate(samples);
      double avg = 0;
      int suc = 0;
      for (const auto &T : mtxs) {
        pcl::PointCloud<pcl::PointXY> pc2 = GenerateData(T);
        auto normals2 = ComputeNormalsPCL(pc2, T);
        NDTMap map2(new LazyGrid2D(2.5));
        map2.loadPointCloud(pc2, normals2);
        map2.computeNDTCells();

        Eigen::Matrix3d Tg = Eigen::Matrix3d::Identity();
        NDTMatcherD2D_2D matcher;
        // matcher.match(map2, map, Tg);
        dprintf("gt (%.2f, %.2f, %.2f)\n", T(0, 2), T(1, 2), acos(T(0, 0)) * 180. / M_PI);
        matcher.ceresmatch(map2, map, Tg);
        Eigen::Matrix3d Tr = Tg;
        avg += diff(Tr, T);
        if (diff(Tr, T) < 3) {
          ++suc;
        }
      }
      avg /= mtxs.size();
      cout << suc << ", " << avg << "\n";
    }
      cout << endl;
  }
#endif

#if 1
  // double x = 4., y = 1., th = 10.;
  double x = 0., y = 3., th = 0.;
  if (argc == 4) {
    x = atof(argv[1]);
    y = atof(argv[2]);
    th = atof(argv[3]);
  }
  Eigen::Matrix3d T = GenTF(x, y, th);
  pcl::PointCloud<pcl::PointXY> pc = GenerateData();
  auto normals = ComputeNormalsPCL(pc);
  NDTMap map(new LazyGrid2D(2.5));
  map.loadPointCloud(pc, normals);
  map.computeNDTCells();

  pcl::PointCloud<pcl::PointXY> pc2 = GenerateData(T);
  auto normals2 = ComputeNormalsPCL(pc2, T);
  NDTMap map2(new LazyGrid2D(2.5));
  map2.loadPointCloud(pc2, normals2);
  map2.computeNDTCells();

  NDTMatcherD2D_2D matcher;
  Eigen::Matrix3d Ttemp = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d Tout0, Tout1;
  matcher.match(map2, map, Ttemp);
  Tout0 = Ttemp;
  Ttemp = Eigen::Matrix3d::Identity();
  matcher.ceresmatch(map2, map, Ttemp);
  Tout1 = Ttemp;
  cout << "result T0: " << endl << Tout0 << endl;
  cout << "result T1: " << endl << Tout1 << endl;
  cout << "actual T: " << endl << T << endl;
  printf("T0: %.2f\n", diff(T, Tout0));
  printf("T1: %.2f\n", diff(T, Tout1));

  pcl::PointCloud<pcl::PointXY> pc3 = GenerateData(Tout0);
  auto normals3 = ComputeNormalsPCL(pc3, Tout0);
  NDTMap map3(new LazyGrid2D(2.5));
  map3.loadPointCloud(pc3, normals3);
  map3.computeNDTCells();

  pcl::PointCloud<pcl::PointXY> pc4 = GenerateData(Tout1);
  auto normals4 = ComputeNormalsPCL(pc4, Tout1);
  NDTMap map4(new LazyGrid2D(2.5));
  map4.loadPointCloud(pc4, normals4);
  map4.computeNDTCells();

  ros::init(argc, argv, "main");
  ros::NodeHandle nh;
  sndt::NDTMapMsg msg, msg2, msg3, msg4;
  toMessage(&map, msg, "map");
  toMessage(&map2, msg2, "map");
  toMessage(&map3, msg3, "map");
  toMessage(&map4, msg4, "map");
  ros::Publisher pub_map = nh.advertise<sndt::NDTMapMsg>("map", 0, true);
  ros::Publisher pub_map2 = nh.advertise<sndt::NDTMapMsg>("map2", 0, true);
  ros::Publisher pub_map3 = nh.advertise<sndt::NDTMapMsg>("map3", 0, true);
  ros::Publisher pub_map4 = nh.advertise<sndt::NDTMapMsg>("map4", 0, true);
  ros::Publisher pub_pc = nh.advertise<sensor_msgs::PointCloud2>("pc", 0, true);
  ros::Publisher pub_pc2 = nh.advertise<sensor_msgs::PointCloud2>("pc2", 0, true);
  ros::Publisher pub_normal = nh.advertise<visualization_msgs::MarkerArray>("normal", 0, true);
  ros::Publisher pub_normal2 = nh.advertise<visualization_msgs::MarkerArray>("normal2", 0, true);

  pub_map.publish(msg);
  pub_map2.publish(msg2);
  pub_map3.publish(msg3);
  pub_map4.publish(msg4);
  pub_pc.publish(PC3D(pc));
  pub_pc2.publish(PC3D(pc2));
  pub_normal.publish(NormalsMarkerArray(pc, normals));
  pub_normal2.publish(NormalsMarkerArray(pc2, normals2));

  ros::spin();
#endif
}
