/**
 * @file ndt_map.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class Definition of NDTMap
 * @version 0.1
 * @date 2021-07-29
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <sndt/ndt_map.h>

NDTMap::NDTMap(double cell_size) : MapInterface(cell_size) {}

NDTMap::~NDTMap() {
  if (is_loaded_) {
    for (auto &cell : cells_) delete cell;
    for (int i = 0; i < cellptrs_size_(0); ++i) delete[] cellptrs_[i];
    delete[] cellptrs_;
  }
}

void NDTMap::LoadPoints(const std::vector<Eigen::Vector2d> &points) {
  if (is_loaded_) {
    for (auto &cell : cells_) delete cell;
    for (int i = 0; i < cellptrs_size_(0); ++i) delete[] cellptrs_[i];
    delete[] cellptrs_;
  }
  GuessMapSize(points);
  Initialize();
  std::vector<NDTCell *> update_cells;
  for (size_t i = 0; i < points.size(); ++i) {
    NDTCell *cell = GetCellAndAllocate(points[i]);
    if (cell) {
      cell->AddPoint(points[i]);
      update_cells.push_back(cell);
      points_.push_back(points[i]);
    }
  }
  for (auto cell : update_cells) cell->ComputeGaussian();
  is_loaded_ = true;
}

void NDTMap::LoadPointsWithCovariances(
    const std::vector<Eigen::Vector2d> &points,
    const std::vector<Eigen::Matrix2d> &point_covs) {
  if (points.size() != point_covs.size()) {
    std::cerr << __FUNCTION__ << ": unmatched sizes\n";
    std::exit(1);
  }
  if (is_loaded_) {
    for (auto &cell : cells_) delete cell;
    for (int i = 0; i < cellptrs_size_(0); ++i) delete[] cellptrs_[i];
    delete[] cellptrs_;
  }
  GuessMapSize(points);
  Initialize();
  std::vector<NDTCell *> update_cells;
  for (size_t i = 0; i < points.size(); ++i) {
    NDTCell *cell = GetCellAndAllocate(points[i]);
    if (cell) {
      cell->AddPointWithCovariance(points[i], point_covs[i]);
      update_cells.push_back(cell);
      points_.push_back(points[i]);
    }
  }
  for (auto cell : update_cells) cell->ComputeGaussian();
  is_loaded_ = true;
}

std::vector<std::shared_ptr<NDTCell>> NDTMap::PseudoTransformCells(
    const Eigen::Affine2d &T, bool include_data) const {
  std::vector<std::shared_ptr<NDTCell>> ret;
  Eigen::Matrix2d R = T.rotation();
  double skew_rad = Eigen::Rotation2Dd(R).angle();
  for (auto it = begin(); it != end(); ++it) {
    auto cell = std::make_shared<NDTCell>();
    cell->SetN((*it)->GetN());
    cell->SetPHasGaussian((*it)->GetPHasGaussian());
    cell->SetSkewRad(skew_rad);
    cell->SetCenter(T * (*it)->GetCenter());
    cell->SetSize((*it)->GetSize());
    cell->SetPointMean(T * (*it)->GetPointMean());
    if ((*it)->GetPHasGaussian()) {
      cell->SetPointCov(R * (*it)->GetPointCov() * R.transpose());
      cell->SetPointEvals((*it)->GetPointEvals());
      cell->SetPointEvecs(R * (*it)->GetPointEvecs());
    }
    if (include_data) {
      cell->SetPoints(TransformPoints((*it)->GetPoints(), T));
    }
    ret.push_back(cell);
  }
  return ret;
}

NDTCell *NDTMap::GetCellAndAllocate(const Eigen::Vector2d &point) {
  if (!point.allFinite()) {
    return nullptr;
  }
  auto idx = GetIndexForPoint(point);
  if (!IsValidIndex(idx)) {
    return nullptr;
  }
  if (!cellptrs_[idx(0)][idx(1)]) {
    cellptrs_[idx(0)][idx(1)] = new NDTCell();
    Eigen::Vector2d center;
    center(0) = map_center_(0) + (idx(0) - map_center_index_(0)) * cell_size_;
    center(1) = map_center_(1) + (idx(1) - map_center_index_(1)) * cell_size_;
    cellptrs_[idx(0)][idx(1)]->SetCenter(center);
    cellptrs_[idx(0)][idx(1)]->SetSize(cell_size_);
    cell_idx_map_[cellptrs_[idx(0)][idx(1)]] = cells_.size();
    cells_.push_back(cellptrs_[idx(0)][idx(1)]);
  }
  return cellptrs_[idx(0)][idx(1)];
}

const NDTCell *NDTMap::GetCellForPoint(const Eigen::Vector2d &point) const {
  if (!is_loaded_) {
    return nullptr;
  }
  auto idx = GetIndexForPoint(point);
  if (!IsValidIndex(idx)) {
    return nullptr;
  }
  return cellptrs_[idx(0)][idx(1)];
}

std::string NDTMap::ToString() const {
  char c[600];
  sprintf(c,
          "cell size: %.2f\n"
          "map center: (%.2f, %.2f)\n"
          "map size: (%.2f, %.2f)\n"
          "cellptrs size: (%d, %d)\n"
          "center index: (%d, %d)\n"
          "active cells: %ld\n",
          cell_size_, map_center_(0), map_center_(1), map_size_(0),
          map_size_(1), cellptrs_size_(0), cellptrs_size_(1),
          map_center_index_(0), map_center_index_(1), cells_.size());
  std::string ret(c);
  for (size_t i = 0; i < cells_.size(); ++i) {
    auto idx = GetIndexForPoint(cells_[i]->GetCenter());
    char s[30];
    sprintf(s, "[#%ld, (%d, %d)] ", i, idx(0), idx(1));
    ret += std::string(s) + cells_[i]->ToString();
  }
  return ret;
}

void NDTMap::ShowCellDistri() const {
  int noinit = 0, nopts = 0, valid = 0, rescale = 0, invalid = 0, one = 0,
      two = 0, gau = 0;
  for (auto cell : cells_) {
    if (cell->GetCellType() == NDTCell::kNoInit) ++noinit;
    if (cell->GetCellType() == NDTCell::kNoPoints) ++nopts;
    if (cell->GetCellType() == NDTCell::kRegular) ++valid;
    if (cell->GetCellType() == NDTCell::kRescale) ++rescale;
    if (cell->GetCellType() == NDTCell::kInvalid) ++invalid;
    if (cell->GetN() == 1) ++one;
    if (cell->GetN() == 2) ++two;
    if (cell->HasGaussian()) ++gau;
  }
  ::printf("%ld cells: %d one, %d two, %d gau\n", cells_.size(), one, two, gau);
  ::printf("nin: %d, npt: %d, reg: %d, res: %d, inv: %d\n\n", noinit, nopts,
           valid, rescale, invalid);
}

std::vector<Eigen::Vector2d> NDTMap::GetPoints() const { return points_; }

std::vector<Eigen::Vector2d> NDTMap::GetPointsWithGaussianCell() const {
  std::vector<Eigen::Vector2d> ret;
  for (auto it = begin(); it != end(); ++it)
    if ((*it)->HasGaussian())
      for (auto pt : (*it)->GetPoints()) ret.push_back(pt);
  return ret;
}

/**
 * XXX: there should exists better solution
 *   1. const_cast is required to match the type of unordered_map key
 *   2. at() rather than [] is used to return an rvalue, prevent CE
 *      since the funciton is called by another const function
 * TODO: probable solution
 *   I may add a field called index or something in SNDTCell
 */
int NDTMap::GetCellIndex(const NDTCell *cell) const {
  auto c = const_cast<NDTCell *>(cell);
  if (!cell_idx_map_.count(c)) {
    std::cerr << __FUNCTION__ << ": not find query cell\n";
    std::exit(1);
  }
  return cell_idx_map_.at(c);
}

void NDTMap::Initialize() {
  cellptrs_ = new NDTCell **[cellptrs_size_(0)];
  for (int i = 0; i < cellptrs_size_(0); ++i) {
    cellptrs_[i] = new NDTCell *[cellptrs_size_(1)];
    memset(cellptrs_[i], 0, cellptrs_size_(1) * sizeof(NDTCell *));
  }
}
