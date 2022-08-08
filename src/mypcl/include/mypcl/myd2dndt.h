#pragma once

#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/registration/registration.h>
#include <pcl/pcl_macros.h>

namespace pcl {
template <typename PointSource, typename PointTarget>
class NormalDistributionsTransformD2D : public Registration<PointSource, PointTarget> {
protected:
  using SourceGrid = VoxelGridCovariance<PointSource>;
  using TargetGrid = VoxelGridCovariance<PointTarget>;
  using TargetGridLeafConstPtr = typename TargetGrid::LeafConstPtr;
  using PointCloudSource =
      typename Registration<PointSource, PointTarget>::PointCloudSource;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;
  using PointCloudTarget =
      typename Registration<PointSource, PointTarget>::PointCloudTarget;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;
  using Registration<PointSource, PointTarget>::reg_name_;
  using Registration<PointSource, PointTarget>::input_;
  using Registration<PointSource, PointTarget>::target_;
  using Registration<PointSource, PointTarget>::nr_iterations_;
  using Registration<PointSource, PointTarget>::max_iterations_;
  using Registration<PointSource, PointTarget>::previous_transformation_;
  using Registration<PointSource, PointTarget>::final_transformation_;
  using Registration<PointSource, PointTarget>::transformation_;
  using Registration<PointSource, PointTarget>::transformation_epsilon_;
  using Registration<PointSource, PointTarget>::transformation_rotation_epsilon_;
  using Registration<PointSource, PointTarget>::converged_;
  using Registration<PointSource, PointTarget>::update_visualizer_;

  struct TransLeaf {
    explicit TransLeaf(const Eigen::Vector3d& _mean, const Eigen::Matrix3d& _cov)
    : mean(_mean), cov(_cov)
    {}
    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
  };

public:
  using Ptr = shared_ptr<NormalDistributionsTransformD2D<PointSource, PointTarget>>;
  using ConstPtr =
      shared_ptr<const NormalDistributionsTransformD2D<PointSource, PointTarget>>;

  NormalDistributionsTransformD2D();

  ~NormalDistributionsTransformD2D() {}

  inline void
  setInputSource(const PointCloudSourceConstPtr& cloud) override
  {
    Registration<PointSource, PointTarget>::setInputSource(cloud);
    setSourceCells();
  }

  inline void
  setInputTarget(const PointCloudTargetConstPtr& cloud) override
  {
    Registration<PointSource, PointTarget>::setInputTarget(cloud);
    setTargetCells();
  }

  inline void
  setResolution(float resolution)
  {
    if (resolution_ != resolution) {
      resolution_ = resolution;
      if (input_)
        setSourceCells();
      if (target_)
        setTargetCells();
    }
  }

  inline float
  getResolution() const
  {
    return resolution_;
  }

  inline double
  getStepSize() const
  {
    return step_size_;
  }

  inline void
  setStepSize(double step_size)
  {
    step_size_ = step_size;
  }

  inline double
  getOulierRatio() const
  {
    return outlier_ratio_;
  }

  inline void
  setOulierRatio(double outlier_ratio)
  {
    outlier_ratio_ = outlier_ratio;
  }

  inline double
  getTransformationLikelihood() const
  {
    return trans_likelihood_;
  }

  inline int
  getFinalNumIteration() const
  {
    return nr_iterations_;
  }

  static void
  convertTransform(const Eigen::Matrix<double, 6, 1>& x, Eigen::Affine3f& trans)
  {
    trans = Eigen::Translation3f(x.head(3).cast<float>()) *
            Eigen::AngleAxisf(float(x(3)), Eigen::Vector3f::UnitX()) *
            Eigen::AngleAxisf(float(x(4)), Eigen::Vector3f::UnitY()) *
            Eigen::AngleAxisf(float(x(5)), Eigen::Vector3f::UnitZ());
  }

  static void
  convertTransform(const Eigen::Matrix<double, 6, 1>& x, Eigen::Matrix4f& trans)
  {
    Eigen::Affine3f _affine;
    convertTransform(x, _affine);
    trans = _affine.matrix();
  }

protected:
  inline void
  setSourceCells()
  {
    source_cells_.setLeafSize(resolution_, resolution_, resolution_);
    source_cells_.setInputCloud(input_);
    source_cells_.filter(true);
  }

  inline void
  setTargetCells()
  {
    target_cells_.setLeafSize(resolution_, resolution_, resolution_);
    target_cells_.setInputCloud(target_);
    target_cells_.filter(true);
  }

  void
  computeTransformation(PointCloudSource& output,
                        const Eigen::Matrix4f& guess) override;

  double
  computeDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                     Eigen::Matrix<double, 6, 6>& hessian,
                     const std::vector<TransLeaf>& trans_leaves,
                     const Eigen::Matrix<double, 6, 1>& transform,
                     bool compute_hessian = true);

  double
  updateDerivatives(Eigen::Matrix<double, 6, 1>& score_gradient,
                    Eigen::Matrix<double, 6, 6>& hessian,
                    const Eigen::Vector3d& x_trans,
                    const Eigen::Matrix3d& c_inv,
                    bool compute_hessian = true) const;

  void
  computeRotationDerivatives(const Eigen::Matrix<double, 6, 1>& transform,
                             bool compute_hessian = true);

  void
  computeCellDerivatives(const Eigen::Vector3d& mean,
                         const Eigen::Matrix3d& cov,
                         bool compute_hessian = true);

  void
  computeHessian(Eigen::Matrix<double, 6, 6>& hessian,
                 const std::vector<TransLeaf>& trans_leaves);

  void
  updateHessian(Eigen::Matrix<double, 6, 6>& hessian,
                const Eigen::Vector3d& uij,
                const Eigen::Matrix3d& B) const;

  double
  computeStepLengthMT(const Eigen::Matrix<double, 6, 1>& transform,
                      Eigen::Matrix<double, 6, 1>& step_dir,
                      double step_init,
                      double step_max,
                      double step_min,
                      double& score,
                      Eigen::Matrix<double, 6, 1>& score_gradient,
                      Eigen::Matrix<double, 6, 6>& hessian,
                      PointCloudSource& trans_cloud);

  bool
  updateIntervalMT(double& a_l,
                   double& f_l,
                   double& g_l,
                   double& a_u,
                   double& f_u,
                   double& g_u,
                   double a_t,
                   double f_t,
                   double g_t) const;

  double
  trialValueSelectionMT(double a_l,
                        double f_l,
                        double g_l,
                        double a_u,
                        double f_u,
                        double g_u,
                        double a_t,
                        double f_t,
                        double g_t) const;

  inline double
  auxilaryFunction_PsiMT(
      double a, double f_a, double f_0, double g_0, double mu = 1.e-4) const
  {
    return f_a - f_0 - mu * g_0 * a;
  }

  inline double
  auxilaryFunction_dPsiMT(double g_a, double g_0, double mu = 1.e-4) const
  {
    return g_a - mu * g_0;
  }

  TargetGrid target_cells_;
  SourceGrid source_cells_;
  float resolution_;
  double step_size_;
  double outlier_ratio_;
  double d1_, d2_;
  double trans_likelihood_;

  Eigen::Matrix<double, 9, 3> dR_;
  Eigen::Matrix<double, 18, 3> ddR_;
  Eigen::Matrix<double, 3, 6> jas_;
  Eigen::Matrix<double, 3, 18> Zas_;
  Eigen::Matrix<double, 18, 6> Habs_;
  Eigen::Matrix<double, 18, 18> Zabs_;
  Eigen::Matrix3d R_;
  Eigen::Vector3d t_;

public:
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace pcl

// #include <myd2dndt.hpp>

namespace pcl {

template <typename PointSource, typename PointTarget>
NormalDistributionsTransformD2D<PointSource,
                                PointTarget>::NormalDistributionsTransformD2D()
: resolution_(1.f)
, step_size_(0.1)
, outlier_ratio_(0.55)
, d1_()
, d2_()
, trans_likelihood_()
{
  reg_name_ = "NormalDistributionsTransformD2D";
  const double gauss_c1 = 10.0 * (1 - outlier_ratio_);
  const double gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
  const double gauss_d3 = -std::log(gauss_c2);
  d1_ = -std::log(gauss_c1 + gauss_c2) - gauss_d3;
  d2_ =
      -2 * std::log((-std::log(gauss_c1 * std::exp(-0.5) + gauss_c2) - gauss_d3) / d1_);

  transformation_epsilon_ = 0.1;
  max_iterations_ = 35;
}

template <typename PointSource, typename PointTarget>
void
NormalDistributionsTransformD2D<PointSource, PointTarget>::computeTransformation(
    PointCloudSource& output, const Eigen::Matrix4f& guess)
{
  nr_iterations_ = 0;
  converged_ = false;

  std::vector<TransLeaf> trans_leaves;
  for (const auto& elem : source_cells_.getLeaves()) {
    if (elem.second.nr_points < 6)
      continue;
    trans_leaves.push_back(TransLeaf(elem.second.mean_, elem.second.cov_));
  }

  Eigen::Affine3d eig_transform = Eigen::Affine3d::Identity();
  if (guess != Eigen::Matrix4f::Identity()) {
    final_transformation_ = guess;
    eig_transform = Eigen::Affine3d(final_transformation_.template cast<double>());
    for (auto& leaf : trans_leaves) {
      transformPoint(leaf.mean, leaf.mean, eig_transform);
      leaf.cov =
          eig_transform.rotation() * leaf.cov * eig_transform.rotation().transpose();
    }
  }

  Eigen::Matrix<double, 6, 1> transform, score_gradient;
  transform.head(3) = eig_transform.translation().cast<double>();
  transform.tail(3) = eig_transform.rotation().eulerAngles(0, 1, 2).cast<double>();

  Eigen::Matrix<double, 6, 6> hessian;

  double score = computeDerivatives(score_gradient, hessian, trans_leaves, transform);

  while (!converged_) {
    previous_transformation_ = transformation_;

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(
        hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix<double, 6, 1> delta = sv.solve(-score_gradient);

    double delta_norm = delta.norm();

    if (delta_norm == 0 || std::isnan(delta_norm)) {
      trans_likelihood_ = score / trans_leaves.size();
      converged_ = delta_norm == 0;
      return;
    }

    delta /= delta_norm;
    delta_norm = computeStepLengthMT(transform,
                                     delta,
                                     delta_norm,
                                     step_size_,
                                     transformation_epsilon_ / 2,
                                     score,
                                     score_gradient,
                                     hessian,
                                     output);
    delta *= delta_norm;

    convertTransform(delta, transformation_);

    transform += delta;

    if (update_visualizer_)
      update_visualizer_(output, pcl::Indices(), *target_, pcl::Indices());

    const double cos_angle =
        0.5 * (transformation_.template block<3, 3>(0, 0).trace() - 1);
    const double translation_sqr =
        transformation_.template block<3, 1>(0, 3).squaredNorm();

    nr_iterations_++;

    if (nr_iterations_ >= max_iterations_ ||
        ((transformation_epsilon_ > 0 && translation_sqr <= transformation_epsilon_) &&
         (transformation_rotation_epsilon_ > 0 &&
          cos_angle >= transformation_rotation_epsilon_)) ||
        ((transformation_epsilon_ <= 0) &&
         (transformation_rotation_epsilon_ > 0 &&
          cos_angle >= transformation_rotation_epsilon_)) ||
        ((transformation_epsilon_ > 0 && translation_sqr <= transformation_epsilon_) &&
         (transformation_rotation_epsilon_ <= 0))) {
      converged_ = true;
    }
  }
  trans_likelihood_ = score / trans_leaves.size();
}

template <typename PointSource, typename PointTarget>
double
NormalDistributionsTransformD2D<PointSource, PointTarget>::computeDerivatives(
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    const std::vector<TransLeaf>& trans_leaves,
    const Eigen::Matrix<double, 6, 1>& transform,
    bool compute_hessian)
{
  score_gradient.setZero();
  hessian.setZero();
  double score = 0;
  computeRotationDerivatives(transform);

  for (const auto& leaf : trans_leaves) {
    const auto& mean = leaf.mean;
    const auto& cov = leaf.cov;
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    PointSource meanpt;
    meanpt.x = mean(0), meanpt.y = mean(1), meanpt.z = mean(2);
    target_cells_.radiusSearch(meanpt, resolution_, neighborhood, distances);
    // Nearest Neighbor / Set size 1
    // target_cells_.nearestKSearch(meanpt, 1, neighborhood, distances);
    for (const auto& cell : neighborhood) {
      if (cell->nr_points < 6)
        continue;
      Eigen::Vector3d uij = mean - cell->getMean();
      Eigen::Matrix3d B = (R_ * cov * R_.transpose() + cell->getCov()).inverse();
      computeCellDerivatives(mean, cov, compute_hessian);
      score += updateDerivatives(score_gradient, hessian, uij, B, compute_hessian);
    }
  }
  return score;
}

template <typename PointSource, typename PointTarget>
void
NormalDistributionsTransformD2D<PointSource, PointTarget>::computeRotationDerivatives(
    const Eigen::Matrix<double, 6, 1>& transform, bool compute_hessian)
{
  const auto calculate_cos_sin = [](double angle, double& c, double& s) {
    if (std::abs(angle) < 10e-5) {
      c = 1.0;
      s = 0.0;
    }
    else {
      c = std::cos(angle);
      s = std::sin(angle);
    }
  };
  double cx, cy, cz, sx, sy, sz;
  calculate_cos_sin(transform(3), cx, sx);
  calculate_cos_sin(transform(4), cy, sy);
  calculate_cos_sin(transform(5), cz, sz);

  // clang-format off
  R_.row(0) << cy * cz, -sz * cy, sy;
  R_.row(1) << sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy;
  R_.row(2) << sx * sz - sy * cx * cz, sx * cz + sy * sz * cx, cx * cy;
  t_ = transform.head(3);

  dR_.row(0) << 0, 0, 0;
  dR_.row(1) << -sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy;
  dR_.row(2) << sx * sy * cz + sz * cx, -sx * sy * sz + cx * cz, -sx * cy;
  dR_.row(3) << -sy * cz, sy * sz, cy;
  dR_.row(4) << sx * cy * cz, -sx * sz * cy, sx * sy;
  dR_.row(5) << -cx * cy * cz, sz * cx * cy, -sy * cx;
  dR_.row(6) << -sz * cy, -cy * cz, 0;
  dR_.row(7) << -sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0;
  dR_.row(8) << sx * cz + sy * sz * cx, -sx * sz + sy * cx * cz, 0;

  if (compute_hessian) {
    ddR_.row(0) << 0, 0, 0;
    ddR_.row(1) << -sx * sy * cz - sz * cx, sx * sy * sz - cx * cz, sx * cy;
    ddR_.row(2) << -sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, -cx * cy;
    ddR_.row(3) << 0, 0, 0;
    ddR_.row(4) << cx * cy * cz, -sz * cx * cy, sy * cx;
    ddR_.row(5) << sx * cy * cz, -sx * sz * cy, sx * sy;
    ddR_.row(6) << 0, 0, 0;
    ddR_.row(7) << -sx * cz - sy * sz * cx, sx * sz - sy * cx * cz, 0;
    ddR_.row(8) << -sx * sy * sz + cx * cz, -sx * sy * cz - sz * cx, 0;
    ddR_.row(9) << -cy * cz, sz * cy, -sy;
    ddR_.row(10) << -sx * sy * cz, sx * sy * sz, sx * cy;
    ddR_.row(11) << sy * cx * cz, -sy * sz * cx, -cx * cy;
    ddR_.row(12) << sy * sz, sy * cz, 0;
    ddR_.row(13) << -sx * sz * cy, -sx * cy * cz, 0;
    ddR_.row(14) << sz * cx * cy, cx * cy * cz, 0;
    ddR_.row(15) << -cy * cz, sz * cy, 0;
    ddR_.row(16) << -sx * sy * cz - sz * cx, sx * sy * sz - cx * cz, 0;
    ddR_.row(17) << -sx * sz + sy * cx * cz, -sx * cz - sy * sz * cx, 0;
  }
  // clang-format on
}

template <typename PointSource, typename PointTarget>
void
NormalDistributionsTransformD2D<PointSource, PointTarget>::computeCellDerivatives(
    const Eigen::Vector3d& mean, const Eigen::Matrix3d& cov, bool compute_hessian)
{
  Eigen::Matrix<double, 9, 1> j = dR_ * mean;
  jas_.col(0) = Eigen::Vector3d::UnitX();
  jas_.col(1) = Eigen::Vector3d::UnitY();
  jas_.col(2) = Eigen::Vector3d::UnitZ();
  jas_.col(3) << j.segment(0, 3);
  jas_.col(4) << j.segment(3, 3);
  jas_.col(5) << j.segment(6, 3);

  Eigen::Matrix<double, 9, 3> za = dR_ * cov * R_.transpose();
  Zas_.setZero();
  Zas_.block<3, 3>(0, 9) = za.block<3, 3>(0, 0) + za.block<3, 3>(0, 0).transpose();
  Zas_.block<3, 3>(0, 12) = za.block<3, 3>(3, 0) + za.block<3, 3>(3, 0).transpose();
  Zas_.block<3, 3>(0, 15) = za.block<3, 3>(6, 0) + za.block<3, 3>(6, 0).transpose();

  if (compute_hessian) {
    Eigen::Matrix<double, 18, 1> h = ddR_ * mean;
    Habs_.setZero();
    Habs_.block<3, 1>(9, 3) << h.segment(0, 3);
    Habs_.block<3, 1>(9, 4) << h.segment(3, 3);
    Habs_.block<3, 1>(9, 5) << h.segment(6, 3);
    Habs_.block<3, 1>(12, 4) << h.segment(9, 3);
    Habs_.block<3, 1>(12, 5) << h.segment(12, 3);
    Habs_.block<3, 1>(15, 5) << h.segment(15, 3);
    Habs_.block<3, 1>(12, 3) = Habs_.block<3, 1>(9, 4);
    Habs_.block<3, 1>(15, 3) = Habs_.block<3, 1>(9, 5);
    Habs_.block<3, 1>(15, 4) = Habs_.block<3, 1>(12, 5);

    Eigen::Matrix<double, 18, 3> z = ddR_ * cov * R_.transpose();
    int i = 0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        z.block<3, 3>(i * 3, 0) +=
            Zas_.block<3, 3>(a * 3, 0) * cov * Zas_.block<3, 3>(b * 3, 0).transpose();
        Eigen::Matrix3d tp = z.block<3, 3>(i * 3, 0).transpose();
        z.block<3, 3>(i * 3, 0) += tp;
        ++i;
      }
    }
    Zabs_.setZero();
    Zabs_.block<3, 3>(9, 9) = z.block<3, 3>(0, 0);
    Zabs_.block<3, 3>(9, 12) = Zabs_.block<3, 3>(12, 9) = z.block<3, 3>(3, 0);
    Zabs_.block<3, 3>(9, 15) = Zabs_.block<3, 3>(15, 9) = z.block<3, 3>(6, 0);
    Zabs_.block<3, 3>(12, 12) = z.block<3, 3>(9, 0);
    Zabs_.block<3, 3>(12, 15) = Zabs_.block<3, 3>(15, 12) = z.block<3, 3>(12, 0);
    Zabs_.block<3, 3>(15, 15) = z.block<3, 3>(15, 0);
  }
}

template <typename PointSource, typename PointTarget>
double
NormalDistributionsTransformD2D<PointSource, PointTarget>::updateDerivatives(
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    const Eigen::Vector3d& uij,
    const Eigen::Matrix3d& B,
    bool compute_hessian) const
{
  Eigen::Transpose<const Eigen::Vector3d> uijT(uij);
  double cost = -d1_ * std::exp(-0.5 * d2_ * uijT * B * uij);
  for (int a = 0; a < 6; ++a) {
    Eigen::Ref<const Eigen::Vector3d> ja(jas_.block<3, 1>(0, a));
    Eigen::Ref<const Eigen::Matrix3d> Za(Zas_.block<3, 3>(0, 3 * a));
    double qa = (2 * uijT * B * ja - uijT * B * Za * B * uij)(0);
    score_gradient(a) += -0.5 * d2_ * cost * qa;

    if (compute_hessian) {
      for (int b = 0; b < 6; ++b) {
        Eigen::Ref<const Eigen::Vector3d> jb(jas_.block<3, 1>(0, b));
        Eigen::Transpose<const Eigen::Vector3d> jbT(jas_.block<3, 1>(0, b));
        Eigen::Ref<const Eigen::Matrix3d> Zb(Zas_.block<3, 3>(0, 3 * b));
        Eigen::Ref<const Eigen::Vector3d> Hab(Habs_.block<3, 1>(3 * a, b));
        Eigen::Ref<const Eigen::Matrix3d> Zab(Zabs_.block<3, 3>(3 * a, 3 * b));
        double qb = (2 * uijT * B * jb - uijT * B * Zb * B * uij)(0);
        hessian(a, b) += -d2_ * cost *
                         ((jbT * B * ja - uijT * B * Zb * B * ja + uijT * B * Hab -
                           uijT * B * Za * B * jb + uijT * B * Za * B * Zb * B * uij -
                           0.5 * uijT * B * Zab * B * uij)(0) -
                          0.25 * d2_ * qa * qb);
      }
    }
  }

  return cost;
}

template <typename PointSource, typename PointTarget>
void
NormalDistributionsTransformD2D<PointSource, PointTarget>::computeHessian(
    Eigen::Matrix<double, 6, 6>& hessian, const std::vector<TransLeaf>& trans_leaves)
{
  hessian.setZero();

  for (const auto& leaf : trans_leaves) {
    const auto& mean = leaf.mean;
    const auto& cov = leaf.cov;
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    PointSource meanpt;
    meanpt.x = mean(0), meanpt.y = mean(1), meanpt.z = mean(2);
    target_cells_.radiusSearch(meanpt, resolution_, neighborhood, distances);
    // Nearest Neighbor / Set size 1
    // target_cells_.nearestKSearch(meanpt, 1, neighborhood, distances);
    for (const auto& cell : neighborhood) {
      if (cell->nr_points < 6)
        continue;
      Eigen::Vector3d uij = mean - cell->getMean();
      Eigen::Matrix3d B = (R_ * cov * R_.transpose() + cell->getCov()).inverse();
      computeCellDerivatives(mean, cov);
      updateHessian(hessian, uij, B);
    }
  }
}

template <typename PointSource, typename PointTarget>
void
NormalDistributionsTransformD2D<PointSource, PointTarget>::updateHessian(
    Eigen::Matrix<double, 6, 6>& hessian,
    const Eigen::Vector3d& uij,
    const Eigen::Matrix3d& B) const
{
  Eigen::Transpose<const Eigen::Vector3d> uijT(uij);
  double d1d2exp = d1_ * d2_ * std::exp(-0.5 * d2_ * uijT * B * uij);
  for (int a = 0; a < 6; ++a) {
    Eigen::Ref<const Eigen::Vector3d> ja(jas_.block<3, 1>(0, a));
    Eigen::Ref<const Eigen::Matrix3d> Za(Zas_.block<3, 3>(0, 3 * a));
    double qa = (2 * uijT * B * ja - uijT * B * Za * B * uij)(0);

    for (int b = 0; b < 6; ++b) {
      Eigen::Ref<const Eigen::Vector3d> jb(jas_.block<3, 1>(0, b));
      Eigen::Transpose<const Eigen::Vector3d> jbT(jas_.block<3, 1>(0, b));
      Eigen::Ref<const Eigen::Matrix3d> Zb(Zas_.block<3, 3>(0, 3 * b));
      Eigen::Ref<const Eigen::Vector3d> Hab(Habs_.block<3, 1>(3 * a, b));
      Eigen::Ref<const Eigen::Matrix3d> Zab(Zabs_.block<3, 3>(3 * a, 3 * b));
      double qb = (2 * uijT * B * jb - uijT * B * Zb * B * uij)(0);
      hessian(a, b) +=
          d1d2exp * ((jbT * B * ja - uijT * B * Zb * B * ja + uijT * B * Hab -
                      uijT * B * Za * B * jb + uijT * B * Za * B * Zb * B * uij -
                      0.5 * uijT * B * Zab * B * uij)(0) -
                     0.25 * d2_ * qa * qb);
    }
  }
}

template <typename PointSource, typename PointTarget>
bool
NormalDistributionsTransformD2D<PointSource, PointTarget>::updateIntervalMT(
    double& a_l,
    double& f_l,
    double& g_l,
    double& a_u,
    double& f_u,
    double& g_u,
    double a_t,
    double f_t,
    double g_t) const
{
  // Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More,
  // Thuente 1994]
  if (f_t > f_l) {
    a_u = a_t;
    f_u = f_t;
    g_u = g_t;
    return false;
  }
  // Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More,
  // Thuente 1994]
  if (g_t * (a_l - a_t) > 0) {
    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return false;
  }
  // Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More,
  // Thuente 1994]
  if (g_t * (a_l - a_t) < 0) {
    a_u = a_l;
    f_u = f_l;
    g_u = g_l;

    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return false;
  }
  // Interval Converged
  return true;
}

template <typename PointSource, typename PointTarget>
double
NormalDistributionsTransformD2D<PointSource, PointTarget>::trialValueSelectionMT(
    double a_l,
    double f_l,
    double g_l,
    double a_u,
    double f_u,
    double g_u,
    double a_t,
    double f_t,
    double g_t) const
{
  if (a_t == a_l && a_t == a_u) {
    return a_t;
  }

  // Endpoints condition check [More, Thuente 1994], p.299 - 300
  enum class EndpointsCondition { Case1, Case2, Case3, Case4 };
  EndpointsCondition condition;

  if (a_t == a_l) {
    condition = EndpointsCondition::Case4;
  }
  else if (f_t > f_l) {
    condition = EndpointsCondition::Case1;
  }
  else if (g_t * g_l < 0) {
    condition = EndpointsCondition::Case2;
  }
  else if (std::fabs(g_t) <= std::fabs(g_l)) {
    condition = EndpointsCondition::Case3;
  }
  else {
    condition = EndpointsCondition::Case4;
  }

  switch (condition) {
  case EndpointsCondition::Case1: {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l
    // and g_t Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    const double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    const double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, f_t and
    // g_l Equation 2.4.2 [Sun, Yuan 2006]
    const double a_q =
        a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

    if (std::fabs(a_c - a_l) < std::fabs(a_q - a_l)) {
      return a_c;
    }
    return 0.5 * (a_q + a_c);
  }

  case EndpointsCondition::Case2: {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l
    // and g_t Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    const double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    const double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, g_l and
    // g_t Equation 2.4.5 [Sun, Yuan 2006]
    const double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    if (std::fabs(a_c - a_t) >= std::fabs(a_s - a_t)) {
      return a_c;
    }
    return a_s;
  }

  case EndpointsCondition::Case3: {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l
    // and g_t Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    const double w = std::sqrt(z * z - g_t * g_l);
    const double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    const double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    double a_t_next;

    if (std::fabs(a_c - a_t) < std::fabs(a_s - a_t)) {
      a_t_next = a_c;
    }
    else {
      a_t_next = a_s;
    }

    if (a_t > a_l) {
      return std::min(a_t + 0.66 * (a_u - a_t), a_t_next);
    }
    return std::max(a_t + 0.66 * (a_u - a_t), a_t_next);
  }

  default:
  case EndpointsCondition::Case4: {
    // Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u
    // and g_t Equation 2.4.52 [Sun, Yuan 2006]
    const double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
    const double w = std::sqrt(z * z - g_t * g_u);
    // Equation 2.4.56 [Sun, Yuan 2006]
    return a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w);
  }
  }
}

template <typename PointSource, typename PointTarget>
double
NormalDistributionsTransformD2D<PointSource, PointTarget>::computeStepLengthMT(
    const Eigen::Matrix<double, 6, 1>& x,
    Eigen::Matrix<double, 6, 1>& step_dir,
    double step_init,
    double step_max,
    double step_min,
    double& score,
    Eigen::Matrix<double, 6, 1>& score_gradient,
    Eigen::Matrix<double, 6, 6>& hessian,
    PointCloudSource& trans_cloud)
{
  // Set the value of phi(0), Equation 1.3 [More, Thuente 1994]
  const double phi_0 = -score;
  // Set the value of phi'(0), Equation 1.3 [More, Thuente 1994]
  double d_phi_0 = -(score_gradient.dot(step_dir));

  if (d_phi_0 >= 0) {
    // Not a decent direction
    if (d_phi_0 == 0) {
      return 0;
    }
    // Reverse step direction and calculate optimal step.
    d_phi_0 *= -1;
    step_dir *= -1;
  }

  // The Search Algorithm for T(mu) [More, Thuente 1994]

  const int max_step_iterations = 10;
  int step_iterations = 0;

  // Sufficient decreace constant, Equation 1.1 [More, Thuete 1994]
  const double mu = 1.e-4;
  // Curvature condition constant, Equation 1.2 [More, Thuete 1994]
  const double nu = 0.9;

  // Initial endpoints of Interval I,
  double a_l = 0, a_u = 0;

  // Auxiliary function psi is used until I is determined ot be a closed
  // interval, Equation 2.1 [More, Thuente 1994]
  double f_l = auxilaryFunction_PsiMT(a_l, phi_0, phi_0, d_phi_0, mu);
  double g_l = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  double f_u = auxilaryFunction_PsiMT(a_u, phi_0, phi_0, d_phi_0, mu);
  double g_u = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  // Check used to allow More-Thuente step length calculation to be skipped by
  // making step_min == step_max
  bool interval_converged = (step_max - step_min) < 0, open_interval = true;

  double a_t = step_init;
  a_t = std::min(a_t, step_max);
  a_t = std::max(a_t, step_min);

  Eigen::Matrix<double, 6, 1> x_t = x + step_dir * a_t;

  // Convert x_t into matrix form
  convertTransform(x_t, final_transformation_);

  // New transformed point cloud
  transformPointCloud(*input_, trans_cloud, final_transformation_);

  // D2D-NDT transformed source leaves
  std::vector<TransLeaf> trans_leaves;
  Eigen::Affine3d tf(final_transformation_.template cast<double>());
  for (const auto& elem : source_cells_.getLeaves()) {
    if (elem.second.nr_points < 6)
      continue;
    Eigen::Vector3d trans_mean;
    transformPoint(elem.second.mean_, trans_mean, tf);
    Eigen::Matrix3d trans_cov =
        tf.rotation() * elem.second.cov_ * tf.rotation().transpose();
    trans_leaves.push_back(TransLeaf(trans_mean, trans_cov));
  }

  // Updates score, gradient and hessian.  Hessian calculation is unessisary but
  // testing showed that most step calculations use the initial step suggestion
  // and recalculation the reusable portions of the hessian would intail more
  // computation time.
  score = computeDerivatives(score_gradient, hessian, trans_leaves, x_t, true);

  // Calculate phi(alpha_t)
  double phi_t = -score;
  // Calculate phi'(alpha_t)
  double d_phi_t = -(score_gradient.dot(step_dir));

  // Calculate psi(alpha_t)
  double psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
  // Calculate psi'(alpha_t)
  double d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

  // Iterate until max number of iterations, interval convergance or a value
  // satisfies the sufficient decrease, Equation 1.1, and curvature condition,
  // Equation 1.2 [More, Thuente 1994]
  while (!interval_converged && step_iterations < max_step_iterations &&
         !(psi_t <= 0 /*Sufficient Decrease*/ &&
           d_phi_t <= -nu * d_phi_0 /*Curvature Condition*/)) {
    // Use auxiliary function if interval I is not closed
    if (open_interval) {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    }
    else {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    a_t = std::min(a_t, step_max);
    a_t = std::max(a_t, step_min);

    x_t = x + step_dir * a_t;

    // Convert x_t into matrix form
    convertTransform(x_t, final_transformation_);

    // New transformed point cloud
    // Done on final cloud to prevent wasted computation
    transformPointCloud(*input_, trans_cloud, final_transformation_);

    // D2D-NDT transformed source leaves
    trans_leaves.clear();
    Eigen::Affine3d tf(final_transformation_.template cast<double>());
    for (const auto& elem : source_cells_.getLeaves()) {
      if (elem.second.nr_points < 6)
        continue;
      Eigen::Vector3d trans_mean;
      transformPoint(elem.second.mean_, trans_mean, tf);
      Eigen::Matrix3d trans_cov =
          tf.rotation() * elem.second.cov_ * tf.rotation().transpose();
      trans_leaves.push_back(TransLeaf(trans_mean, trans_cov));
    }

    // Updates score, gradient. Values stored to prevent wasted computation.
    score = computeDerivatives(score_gradient, hessian, trans_leaves, x_t, false);

    // Calculate phi(alpha_t+)
    phi_t = -score;
    // Calculate phi'(alpha_t+)
    d_phi_t = -(score_gradient.dot(step_dir));

    // Calculate psi(alpha_t+)
    psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
    // Calculate psi'(alpha_t+)
    d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

    // Check if I is now a closed interval
    if (open_interval && (psi_t <= 0 && d_psi_t >= 0)) {
      open_interval = false;

      // Converts f_l and g_l from psi to phi
      f_l += phi_0 - mu * d_phi_0 * a_l;
      g_l += mu * d_phi_0;

      // Converts f_u and g_u from psi to phi
      f_u += phi_0 - mu * d_phi_0 * a_u;
      g_u += mu * d_phi_0;
    }

    if (open_interval) {
      // Update interval end points using Updating Algorithm [More, Thuente
      // 1994]
      interval_converged =
          updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    }
    else {
      // Update interval end points using Modified Updating Algorithm [More,
      // Thuente 1994]
      interval_converged =
          updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    step_iterations++;
  }

  // If inner loop was run then hessian needs to be calculated.
  // Hessian is unnessisary for step length determination but gradients are
  // required so derivative and transform data is stored for the next iteration.
  if (step_iterations) {
    computeHessian(hessian, trans_leaves);
  }

  return a_t;
}

} // namespace pcl
