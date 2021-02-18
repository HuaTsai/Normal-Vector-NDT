//
// Created by localuser on 22/07/19.
//



#include <iostream>
#include <thread>
#include <pcl/common/centroid.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <algorithm>
#include <Eigen/Dense>
#include <pcl/common/pca.h>
#include <typeinfo>

#include "normal2d/Normal2dEstimation.h"


int main(){


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr  norm_cloud(new pcl::PointCloud<pcl::Normal>);
    pcl::io::loadPCDFile("/home/ee904/Desktop/HuaTsai/NormalNDT/Research/src/normal2d/sample.pcd", *cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

    Normal2dEstimation norm_estim;
    norm_estim.setInputCloud(cloud);
    norm_estim.setSearchMethod (tree);

     norm_estim.setRadiusSearch (30);

    norm_estim.compute(norm_cloud);

    pcl::visualization::PCLVisualizer viewer;
    viewer.setBackgroundColor (0.0, 0.0, 0.0);
    std::cout << norm_cloud->points.size()<<"  "<<cloud->points.size()<<std::endl;
    viewer.addPointCloud<pcl::PointXYZ>(cloud,"cloud");
    viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, norm_cloud,1.,10.0, "cloud_norm");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_norm");
    while (!viewer.wasStopped ())
    {
        viewer.spinOnce (1);
    }
    return 0;
}


