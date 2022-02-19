#include <sndt/outlier_reject.h>

RangeOutlierRejection::RangeOutlierRejection() : reject_(false), multiplier_(1.5) {}

void RangeOutlierRejection::AddCorrespondence(const Eigen::Vector2d &p,
                                         const Eigen::Vector2d &q) {
  distances_.push_back((p - q).norm());
}

std::vector<int> RangeOutlierRejection::GetIndices() {
  int n = distances_.size();
  std::vector<int> ret;
  if (reject_) {
    auto mean = std::accumulate(distances_.begin(), distances_.end(), 0.) / n;
    double stdev = 0;
    for (int i = 0; i < n; ++i)
      stdev += (distances_[i] - mean) * (distances_[i] - mean);
    stdev = sqrt(stdev / (n - 1));
    for (int i = 0; i < n; ++i)
      if (distances_[i] < mean + multiplier_ * stdev) ret.push_back(i);
  } else {
    for (int i = 0; i < n; ++i) ret.push_back(i);
  }
  return ret;
}

AngleOutlierRejection::AngleOutlierRejection() : reject_(false), multiplier_(1) {}

void AngleOutlierRejection::AddCorrespondence(const Eigen::Vector2d &np,
                                              const Eigen::Vector2d &nq) {
  angles_.push_back(std::acos(np.dot(nq)));
}

std::vector<int> AngleOutlierRejection::GetIndices() {
  int n = angles_.size();
  std::vector<int> ret;
  if (reject_) {
    auto mean = std::accumulate(angles_.begin(), angles_.end(), 0.) / n;
    double sum = 0;
    for (auto ang : angles_) sum += (ang - mean) * (ang - mean);
    double stdev = std::sqrt(sum / (n - 1));
    for (int i = 0; i < n; ++i)
      if (angles_[i] < mean + multiplier_ * stdev) ret.push_back(i);
      // if (angles_[i] < 30. * M_PI / 180.) ret.push_back(i);
  } else {
    for (int i = 0; i < n; ++i) ret.push_back(i);
  }
  return ret;
}