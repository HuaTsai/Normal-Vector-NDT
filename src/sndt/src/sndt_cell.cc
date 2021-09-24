/**
 * @file sndt_cell.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class Definition of SNDTCell
 * @version 0.1
 * @date 2021-07-29
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <sndt/sndt_cell.h>
#include <sndt/eigen_utils.h>
#include <boost/functional/hash.hpp>

SNDTCell::SNDTCell() {
  pcelltype_ = kNoInit;
  ncelltype_ = kNoInit;
  rescale_ratio_ = 100.;
  tolerance_ = Eigen::NumTraits<double>::dummy_precision();
  nhasgaussian_ = false;
  ncov_.setZero();
  nevecs_.setZero();
  nmean_.setZero();
  nevals_.setZero();
}

void SNDTCell::ComputeGaussian() {
  n_ = points_.size();
  ComputePGaussian();
  ComputeNGaussian();
}

void SNDTCell::ComputePGaussian() {
  pmean_.setZero(), pcov_.setZero();
  auto valids = ExcludeNaNInf2(points_, point_covs_);
  if (valids.first.size() == 0) {
    pcelltype_ = kNoPoints;
    return;
  }
  pmean_ = ComputeMean(valids.first);
  pcov_ = ComputeCov(valids.first, pmean_, valids.second);
  if (!pcov_.isZero()) {
    ComputeEvalEvec(pcov_, pevals_, pevecs_);
    if (pevals_(0) <= tolerance_ || pevals_(1) <= tolerance_) {
      pcelltype_ = kInvalid;
      return;
    }
    pcelltype_ = kRegular;
    phasgaussian_ = true;
    if (pevals_(1) > rescale_ratio_ * pevals_(0)) {
      pevals_(0) = pevals_(1) / rescale_ratio_;
      pcov_ = pevecs_ * pevals_.asDiagonal() * pevecs_.transpose();
      pcelltype_ = kRescale;
    }
  }
}

void SNDTCell::ComputeNGaussian() {
  nmean_.setZero(), ncov_.setZero();
  for (auto &nm : normals_)
    if (nm.allFinite() && nm.dot(pevecs_.col(0)) < 0)
      nm = -nm;
  auto valids = ExcludeNaNInf(normals_);
  if (!valids.size()) {
    ncelltype_ = kNoPoints;
    return;
  }
  std::unordered_set<std::pair<int, int>, boost::hash<std::pair<int, int>>> st;
  for (auto valid : valids)
    st.insert({valid(0) * 1e5, valid(1) * 1e5});
  nmean_ = ComputeMean(valids);
  if (st.size() <= 2) {
    nevecs_.col(0) = nmean_.normalized();
    nevecs_.col(1) = Eigen::Vector2d(-nevecs_(1, 0), nevecs_(0, 0));
    nevals_ = Eigen::Vector2d(0.0005, 0.01);
    ncov_ = nevecs_ * nevals_.asDiagonal() * nevecs_.transpose();
    ncelltype_ = kAssign;
    nhasgaussian_ = true;
  } else {
    ncov_ = ComputeCov(valids, nmean_);
    ComputeEvalEvec(ncov_, nevals_, nevecs_);
    if (nevals_(0) <= tolerance_ || nevals_(1) <= tolerance_) {
      ncelltype_ = kInvalid;
      return;
    }
    ncelltype_ = kRegular;
    nhasgaussian_ = true;
    if (nevals_(1) > 1000 * nevals_(0)) {
      nevals_(0) = nevals_(1) / 1000;
      ncov_ = nevecs_ * nevals_.asDiagonal() * nevecs_.transpose();
      ncelltype_ = kRescale;
    }
  }
}
// void SNDTCell::ComputePGaussian() {
//   pmean_.setZero(), pcov_.setZero();
//   auto indices = ExcludeNaNInf3(points_, point_covs_);
//   if (!indices.size()) {
//     pcelltype_ = kNoPoints;
//     return;
//   }
//   ComputeMeanAndCov(points_, point_covs_, indices, pmean_, pcov_);
//   if (pcov_.allFinite()) {
//     ComputeEvalEvec(pcov_, pevals_, pevecs_);
//     if (pevals_(0) <= 0 && pevals_(1) <= 0) {
//       pcelltype_ = kInvalid;
//       return;
//     }
//     pcelltype_ = kRegular;
//     phasgaussian_ = true;
//     int maxidx, minidx;
//     double maxval = pevals_.maxCoeff(&maxidx);
//     double minval = pevals_.minCoeff(&minidx);
//     if (maxval > rescale_ratio_ * minval) {
//       pevals_(minidx) = maxval / rescale_ratio_;
//       pcov_ = pevecs_ * pevals_.asDiagonal() * pevecs_.transpose();
//       pcelltype_ = kRescale;
//     }
//   }
// }

// void SNDTCell::ComputeNGaussian() {
//   nmean_.setZero(), ncov_.setZero();
//   auto indices = ExcludeNaNInf3(normals_);
//   if (!indices.size()) {
//     ncelltype_ = kNoPoints;
//     return;
//   }
//   ComputeMeanAndCov(normals_, indices, nmean_, ncov_);
//   if (ncov_.allFinite()) {
//     ComputeEvalEvec(ncov_, nevals_, nevecs_);
//     if (nevals_(0) <= 0 || nevals_(1) <= 0) {
//       ncelltype_ = kInvalid;
//       return;
//     }
//     ncelltype_ = kRegular;
//     nhasgaussian_ = true;
//     int maxidx, minidx;
//     double maxval = nevals_.maxCoeff(&maxidx);
//     double minval = nevals_.minCoeff(&minidx);
//     if (maxval > rescale_ratio_ * minval) {
//       nevals_(minidx) = maxval / rescale_ratio_;
//       ncov_ = nevecs_ * nevals_.asDiagonal() * nevecs_.transpose();
//       ncelltype_ = kRescale;
//     }
//   }
//   if (ncov_.isZero() && indices.size() >= 3) {
//     ncov_ = Eigen::Matrix2d::Identity() * 0.01;
//     ncelltype_ = kAssign;
//     nhasgaussian_ = true;
//   }
// }
bool SNDTCell::HasGaussian() const {
  return phasgaussian_ && nhasgaussian_;
}

std::string SNDTCell::ToString() const {
  char c[600];
  sprintf(c, "cell @ (%.2f, %.2f):\n"
             "   N: %d\n"
             "  𝓝p: %s, 𝓝n: %s\n"
             "  sz: %.2f\n"
             "skew: %.2f\n"
             "  μp: (%.2f, %.2f)\n"
             "  Σp: (%.4f, %.4f, %.4f, %.4f)\n"
             " evp: (%.2f, %.2f)\n"
             " ecp: (%.2f, %.2f), (%.2f, %.2f)\n"
             "  μn: (%.2f, %.2f)\n"
             "  Σn: (%.4f, %.4f, %.4f, %.4f)\n"
             " evn: (%.2f, %.2f)\n"
             " ecn: (%.2f, %.2f), (%.2f, %.2f)\n",
             center_(0), center_(1), n_,
             phasgaussian_ ? "true" : "false", nhasgaussian_ ? "true" : "false", size_, skew_rad_,
             pmean_(0), pmean_(1), pcov_(0, 0), pcov_(0, 1), pcov_(1, 0), pcov_(1, 1),
             pevals_(0), pevals_(1), pevecs_(0, 0), pevecs_(1, 0), pevecs_(1, 0), pevecs_(1, 1),
             nmean_(0), nmean_(1), ncov_(0, 0), ncov_(0, 1), ncov_(1, 0), ncov_(1, 1),
             nevals_(0), nevals_(1), nevecs_(0, 0), nevecs_(1, 0), nevecs_(1, 0), nevecs_(1, 1));
  std::stringstream ss;
  ss.setf(std::ios::fixed);
  ss.precision(2);
  for (int i = 0; i < n_; ++i) {
    ss << "p[" << i << "]: (" << points_[i](0) << ", " << points_[i](1) << "), "
       << "n[" << i << "]: (" << normals_[i](0) << ", " << normals_[i](1) << ")"
       << std::endl;
  }
  return std::string(c) + ss.str();
}
