#include <normal2d/normal2d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <ros/package.h>

using namespace std;
using namespace Eigen;

int main() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr nmpc(new pcl::PointCloud<pcl::Normal>);
  pcl::io::loadPCDFile(ros::package::getPath("normal2d") + "/sample2.pcd", *pc);

  vector<Vector2d> pts;
  for (auto pt : *pc) pts.push_back(Vector2d(pt.x, pt.y));
  auto nms = ComputeNormals(pts, 5);
  for (auto nm : nms) nmpc->push_back(pcl::Normal(nm(0), nm(1), 0));

  pcl::visualization::PCLVisualizer viewer;
  viewer.setBackgroundColor(0.0, 0.0, 0.0);
  viewer.addPointCloud<pcl::PointXYZ>(pc, "pc");
  viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(pc, nmpc, 1., 10.0,
                                                          "nm");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_norm");
  while (!viewer.wasStopped()) viewer.spinOnce(1);
}
