#include <common/other_utils.h>
#include <sndt/outlier_reject.h>

RangeOutlierRejection::RangeOutlierRejection()
    : reject_(false), multiplier_(1.5) {}

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

AngleOutlierRejection::AngleOutlierRejection()
    : reject_(false), multiplier_(1) {}

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

std::vector<int> ThresholdRejection(const std::vector<double> &vals,
                                    double threshold) {
  std::vector<int> ret;
  for (size_t i = 0; i < vals.size(); ++i)
    if (vals[i] < threshold) ret.push_back(i);
  return ret;
}

std::vector<int> StatisticRejection(const std::vector<double> &vals,
                                    double multiplier) {
  auto ms = ComputeMeanAndStdev(vals);
  std::vector<int> ret;
  for (size_t i = 0; i < vals.size(); ++i)
    if (vals[i] < ms.first + multiplier * ms.second) ret.push_back(i);
  return ret;
}

std::vector<int> BothRejection(const std::vector<double> &vals,
                               double multiplier,
                               double threshold) {
  auto ms = ComputeMeanAndStdev(vals);
  std::vector<int> ret;
  for (size_t i = 0; i < vals.size(); ++i)
    if (vals[i] < std::max(ms.first + multiplier * ms.second, threshold))
      ret.push_back(i);
  return ret;
}

std::vector<int> CommonIndices(const std::vector<int> &a,
                               const std::vector<int> &b) {
  std::vector<int> ret;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] == b[j]) {
      ret.push_back(a[i]);
      ++i;
      ++j;
    } else if (a[i] < b[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return ret;
}

void OutlierRejectionMaker::RangeRejection(
    const std::vector<Eigen::Vector2d> &ps,
    const std::vector<Eigen::Vector2d> &qs,
    Rejection method,
    const std::vector<double> &params) {
  std::vector<double> vals;
  for (size_t i = 0; i < ps.size(); ++i) vals.push_back((ps[i] - qs[i]).norm());

  std::vector<int> indices;
  if (method == Rejection::kThreshold)
    indices = ThresholdRejection(vals, params[0]);
  else if (method == Rejection::kStatistic)
    indices = StatisticRejection(vals, params[0]);
  else if (method == Rejection::kBoth)
    indices = BothRejection(vals, params[0], params[1]);

  indices_ = CommonIndices(indices_, indices);
}

void OutlierRejectionMaker::AngleRejection(
    const std::vector<Eigen::Vector2d> &nps,
    const std::vector<Eigen::Vector2d> &nqs,
    Rejection method,
    const std::vector<double> &params) {
  std::vector<double> vals;
  for (size_t i = 0; i < nps.size(); ++i)
    vals.push_back(std::acos(nps[i].dot(nqs[i])));

  std::vector<int> indices;
  if (method == Rejection::kThreshold)
    indices = ThresholdRejection(vals, params[0]);
  else if (method == Rejection::kStatistic)
    indices = StatisticRejection(vals, params[0]);
  else if (method == Rejection::kBoth)
    indices = BothRejection(vals, params[0], params[1]);

  indices_ = CommonIndices(indices_, indices);
}
