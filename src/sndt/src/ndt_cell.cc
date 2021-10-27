/**
 * @file ndt_cell.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class Definition of NDTCell
 * @version 0.1
 * @date 2021-07-29
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include <sndt/ndt_cell.h>
#include <sndt/helpers.h>

NDTCell::NDTCell() {
  celltype_ = kNoInit;
  rescale_ratio_ = 100.;
  tolerance_ = Eigen::NumTraits<double>::dummy_precision();
}

void NDTCell::ComputeGaussian() {
  n_ = points_.size();
  pmean_.setZero(), pcov_.setZero();
  auto valids = ExcludeNaNInf2(points_, point_covs_);
  if (valids.first.size() == 0) {
    celltype_ = kNoPoints;
    return;
  }
  pmean_ = ComputeMean(valids.first);
  pcov_ = ComputeCov(valids.first, pmean_, valids.second);
  if (!pcov_.isZero()) {
    ComputeEvalEvec(pcov_, pevals_, pevecs_);
    if (pevals_(0) <= tolerance_ || pevals_(1) <= tolerance_) {
      celltype_ = kInvalid;
      return;
    }
    celltype_ = kRegular;
    phasgaussian_ = true;
    if (pevals_(1) > rescale_ratio_ * pevals_(0)) {
      pevals_(0) = pevals_(1) / rescale_ratio_;
      pcov_ = pevecs_ * pevals_.asDiagonal() * pevecs_.transpose();
      celltype_ = kRescale;
    }
  }
}

bool NDTCell::HasGaussian() const {
  return phasgaussian_;
}

std::string NDTCell::ToString() const {
  char c[600];
  sprintf(c, "cell @ (%.2f, %.2f):\n"
             "   N: %d\n"
             "  𝓝p: %s\n"
             "  sz: %.2f\n"
             "skew: %.2f\n"
             "  μp: (%.2f, %.2f)\n"
             "  Σp: (%.4f, %.4f, %.4f, %.4f)\n"
             " evp: (%.2f, %.2f)\n"
             " ecp: (%.2f, %.2f), (%.2f, %.2f)\n",
             center_(0), center_(1), n_,
             phasgaussian_ ? "true" : "false", size_, skew_rad_,
             pmean_(0), pmean_(1), pcov_(0, 0), pcov_(0, 1), pcov_(1, 0), pcov_(1, 1),
             pevals_(0), pevals_(1), pevecs_(0, 0), pevecs_(1, 0), pevecs_(1, 0), pevecs_(1, 1));
  std::stringstream ss;
  ss.setf(std::ios::fixed);
  ss.precision(2);
  for (int i = 0; i < n_; ++i)
    ss << "p[" << i << "]: (" << points_[i](0) << ", " << points_[i](1) << ")\n";
  return std::string(c) + ss.str();
}
