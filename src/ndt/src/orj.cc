
#include <common/other_utils.h>
#include <ndt/orj.h>

namespace {
std::vector<int> ThresholdRejection(const std::vector<double> &vals,
                                    double threshold) {
  std::vector<int> ret;
  for (size_t i = 0; i < vals.size(); ++i)
    if (vals[i] < threshold) ret.push_back(i);
  return ret;
}

// TODO: add configurable parameter
std::vector<int> StatisticRejection(const std::vector<double> &vals) {
  int n = vals.size();
  std::vector<std::reference_wrapper<const double>> v(vals.begin(), vals.end());
  std::sort(v.begin(), v.end());
  double threshold = v[n * 3 / 4];
  std::vector<int> ret;
  for (size_t i = 0; i < vals.size(); ++i)
    if (vals[i] < threshold) ret.push_back(i);
  return ret;
}

std::vector<int> BothRejection(const std::vector<double> &vals,
                               double threshold) {
  int n = vals.size();
  std::vector<std::reference_wrapper<const double>> v(vals.begin(), vals.end());
  std::sort(v.begin(), v.end());
  double threshold2 = v[n * 3 / 4];
  std::vector<int> ret;
  for (size_t i = 0; i < vals.size(); ++i)
    if (vals[i] < std::max(threshold, threshold2)) ret.push_back(i);
  return ret;
}

std::vector<int> CommonIndices(const std::vector<int> &a,
                               const std::vector<int> &b) {
  std::vector<int> ret;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] == b[j]) {
      ret.push_back(a[i]);
      ++i, ++j;
    } else if (a[i] < b[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return ret;
}
}  // namespace

void Orj::RangeRejection(const std::vector<Eigen::Vector3d> &ps,
                         const std::vector<Eigen::Vector3d> &qs,
                         Rejection method,
                         const std::vector<double> &params) {
  std::vector<double> vals;
  for (size_t i = 0; i < ps.size(); ++i) vals.push_back((ps[i] - qs[i]).norm());

  std::vector<int> indices;
  if (method == Rejection::kThreshold)
    indices = ThresholdRejection(vals, params[0]);
  else if (method == Rejection::kStatistic)
    indices = StatisticRejection(vals);
  else if (method == Rejection::kBoth)
    indices = BothRejection(vals, params[0]);

  indices_ = CommonIndices(indices_, indices);
}

void Orj::AngleRejection(const std::vector<Eigen::Vector3d> &nps,
                         const std::vector<Eigen::Vector3d> &nqs,
                         Rejection method,
                         const std::vector<double> &params) {
  std::vector<double> vals;
  for (size_t i = 0; i < nps.size(); ++i)
    vals.push_back(std::acos(nps[i].dot(nqs[i])));

  std::vector<int> indices;
  if (method == Rejection::kThreshold)
    indices = ThresholdRejection(vals, params[0]);
  else if (method == Rejection::kStatistic)
    indices = StatisticRejection(vals);
  else if (method == Rejection::kBoth)
    indices = BothRejection(vals, params[0]);

  indices_ = CommonIndices(indices_, indices);
}

void Orj2D::RangeRejection(const std::vector<Eigen::Vector2d> &ps,
                           const std::vector<Eigen::Vector2d> &qs,
                           Rejection method,
                           const std::vector<double> &params) {
  std::vector<double> vals;
  for (size_t i = 0; i < ps.size(); ++i) vals.push_back((ps[i] - qs[i]).norm());

  std::vector<int> indices;
  if (method == Rejection::kThreshold)
    indices = ThresholdRejection(vals, params[0]);
  else if (method == Rejection::kStatistic)
    indices = StatisticRejection(vals);
  else if (method == Rejection::kBoth)
    indices = BothRejection(vals, params[0]);

  indices_ = CommonIndices(indices_, indices);
}

void Orj2D::AngleRejection(const std::vector<Eigen::Vector2d> &nps,
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
    indices = StatisticRejection(vals);
  else if (method == Rejection::kBoth)
    indices = BothRejection(vals, params[0]);

  indices_ = CommonIndices(indices_, indices);
}
