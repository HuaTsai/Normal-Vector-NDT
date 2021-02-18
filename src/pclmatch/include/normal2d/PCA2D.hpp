//
// Created by Francois Gauthier-Clerc on 02/08/19.
//
#pragma once

#include <pcl/PointIndices.h>
#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/search.h>

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point>::Ptr PtrCloud;
typedef pcl::PointCloud<Point>::ConstPtr ConstPtrCloud;
typedef pcl::search::Search<Point>::Ptr PtrkdTree;
typedef pcl::IndicesPtr IndicesPtr;

class PCA2D {
 public:
  PCA2D()
      : m_computed(false), m_indices(new pcl::Indices), m_cloud(nullptr) {}

  void initCompute() {
    if (!m_cloud) {
      throw std::runtime_error(
          "You have to set a cloud before ask any result !");
    }

    pcl::CentroidPoint<Point> centroid;
    // Compute mean
    m_mean = Eigen::Vector2f::Zero();
    Eigen::Vector4f temp_mean;
    compute3DCentroid(*m_cloud, *m_indices, temp_mean);
    m_mean(0) = temp_mean(0);
    m_mean(1) = temp_mean(1);

    // Compute demeanished cloud
    Eigen::MatrixXf cloud_demean;
    demeanPointCloud(*m_cloud, *m_indices, temp_mean, cloud_demean);
    assert(cloud_demean.cols() == int(m_indices->size()));
    // Compute the product cloud_demean * cloud_demean^T
    Eigen::Matrix2f alpha = static_cast<Eigen::Matrix2f>(
        cloud_demean.topRows<2>() * cloud_demean.topRows<2>().transpose());

    // Compute eigen vectors and values
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> evd(alpha);
    // Organize eigenvectors and eigenvalues in ascendent order
    for (int i = 0; i < 2; ++i) {
      m_eigen_values[i] = evd.eigenvalues()[1 - i];
      m_eigen_vectors.col(i) = evd.eigenvectors().col(1 - i);
    }
    // If not basis only then compute the coefficients

    m_computed = true;
  }

  void setInputCloud(const ConstPtrCloud& cloud) {
    m_computed = false;
    m_cloud = cloud;
  }

  void setIndices(const pcl::PointIndices& indices) {
    m_computed = false;
    m_indices->clear();
    m_indices->insert(m_indices->begin(), indices.indices.cbegin(),
                      indices.indices.cend());
  }

  void setIndices(const pcl::PointIndicesPtr& indices) {
    m_computed = false;
    m_indices->insert(m_indices->begin(), indices->indices.cbegin(),
                      indices->indices.cend());
  }

  void setIndices(const pcl::PointIndicesConstPtr& indices) {
    m_computed = false;
    m_indices->insert(m_indices->begin(), indices->indices.cbegin(),
                      indices->indices.cend());
  }

  void setIndices(const boost::shared_ptr<std::vector<int>>& indices) {
    m_computed = false;
    m_indices->insert(m_indices->begin(), indices->cbegin(), indices->cend());
  }

  void setIndices(const std::vector<int>& indices) {
    m_computed = false;
    m_indices->insert(m_indices->begin(), indices.cbegin(), indices.cend());
  }

  Eigen::Vector2f getMean() {
    if (!m_computed) this->initCompute();
    return m_mean;
  }

  Eigen::Matrix2f getEigenVectors() {
    if (!m_computed) this->initCompute();
    return m_eigen_vectors;
  }

  Eigen::Vector2f getEigenValues() {
    if (!m_computed) this->initCompute();
    return m_eigen_values;
  }

  void project(const Point& input, Point& projection) {
    Eigen::Vector2f demean_input = {input.x - m_mean(0), input.y - m_mean(1)};
    auto proj_result = m_eigen_vectors.transpose() * demean_input;
    projection.x = proj_result(0);
    projection.y = proj_result(1);
    projection.z = 0;
  }

  void project(const ConstPtrCloud& in_cloud, PtrCloud& out_cloud) {
    if (in_cloud->is_dense) {
      out_cloud->resize(in_cloud->size());
      for (size_t i = 0; i < in_cloud->size(); ++i)
        this->project(in_cloud->points[i], out_cloud->points[i]);
    } else {
      pcl::PointXYZ p;
      for (size_t i = 0; i < in_cloud->size(); ++i) {
        if (!pcl_isfinite(in_cloud->points[i].x) ||
            !pcl_isfinite(in_cloud->points[i].y) ||
            !pcl_isfinite(in_cloud->points[i].z))
          continue;
        project(in_cloud->points[i], p);
        out_cloud->points.push_back(p);
      }
    }
  }

 private:
  bool m_computed;        ///< True if PCA have been performed, false otherwise.
  IndicesPtr m_indices;   ///< Indice to use with input cloud.
  ConstPtrCloud m_cloud;  ///< InputCloud to deal with.

  Eigen::Vector2f m_mean;           ///< Centroid of input cloud
  Eigen::Matrix2f m_eigen_vectors;  ///< Eigen vector in matrix form.
  Eigen::Vector2f m_eigen_values;   ///< Eigen scalar in vector.
};
