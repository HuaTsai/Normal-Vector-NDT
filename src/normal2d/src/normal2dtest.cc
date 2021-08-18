// Created by Francois Gauthier-Clerc.
// Modified and refactored by HuaTsai.
#include <normal2d/normal2d.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <ros/package.h>

int main() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr norm_cloud(
      new pcl::PointCloud<pcl::Normal>);
  pcl::io::loadPCDFile(ros::package::getPath("normal2d") + "/sample.pcd", *cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>);

  Normal2dEstimation norm_estim;
  norm_estim.setInputCloud(cloud);
  norm_estim.setSearchMethod(tree);
  norm_estim.setRadiusSearch(30);
  norm_estim.compute(norm_cloud);

  pcl::visualization::PCLVisualizer viewer;
  viewer.setBackgroundColor(0.0, 0.0, 0.0);
  viewer.addPointCloud<pcl::PointXYZ>(cloud, "cloud");
  viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, norm_cloud, 1.,
                                                          10.0, "cloud_norm");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_norm");
  while (!viewer.wasStopped()) {
    viewer.spinOnce(1);
  }
}
