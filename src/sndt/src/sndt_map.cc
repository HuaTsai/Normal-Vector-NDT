/**
 * @file sndt_map.cc
 * @author HuaTsai (huatsai.eed07g@nctu.edu.tw)
 * @brief Class Definition of SNDTMap
 * @version 0.1
 * @date 2021-07-29
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <sndt/sndt_map.h>

SNDTMap::SNDTMap(double cell_size) : MapInterface(cell_size) {}

SNDTMap::~SNDTMap() {
  if (is_loaded_) {
    for (auto &cell : cells_) delete cell;
    for (int i = 0; i < cellptrs_size_(0); ++i) delete[] cellptrs_[i];
    delete[] cellptrs_;
  }
}

void SNDTMap::LoadPointsAndNormals(
    const std::vector<Eigen::Vector2d> &points,
    const std::vector<Eigen::Vector2d> &normals) {
  if (is_loaded_) {
    for (auto &cell : cells_) delete cell;
    for (int i = 0; i < cellptrs_size_(0); ++i) delete[] cellptrs_[i];
    delete[] cellptrs_;
  }
  GuessMapSize(points);
  Initialize();
  std::vector<SNDTCell *> update_cells;
  for (size_t i = 0; i < points.size(); ++i) {
    SNDTCell *cell = GetCellAndAllocate(points[i]);
    if (cell) {
      cell->AddPoint(points[i]);
      cell->AddNormal(normals[i]);
      update_cells.push_back(cell);
    }
  }
  for (auto cell : update_cells) cell->ComputeGaussian();
  is_loaded_ = true;
}

void SNDTMap::LoadPointsWithCovariancesAndNormals(
    const std::vector<Eigen::Vector2d> &points,
    const std::vector<Eigen::Matrix2d> &point_covs,
    const std::vector<Eigen::Vector2d> &normals) {
  if (!(points.size() == point_covs.size() &&
        points.size() == normals.size())) {
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
  std::vector<SNDTCell *> update_cells;
  for (size_t i = 0; i < points.size(); ++i) {
    SNDTCell *cell = GetCellAndAllocate(points[i]);
    if (cell) {
      cell->AddPointWithCovariance(points[i], point_covs[i]);
      cell->AddNormal(normals[i]);
      update_cells.push_back(cell);
    }
  }
  for (auto cell : update_cells) cell->ComputeGaussian();
  is_loaded_ = true;
}

std::vector<std::shared_ptr<SNDTCell>> SNDTMap::PseudoTransformCells(
    const Eigen::Affine2d &T, bool include_data) const {
  std::vector<std::shared_ptr<SNDTCell>> ret;
  Eigen::Matrix2d R = T.rotation();
  Eigen::Vector2d t = T.translation();
  double skew_rad = Eigen::Rotation2Dd(R).angle();
  for (auto it = begin(); it != end(); ++it) {
    auto cell = std::make_shared<SNDTCell>();
    cell->SetN((*it)->GetN());
    cell->SetPHasGaussian((*it)->GetPHasGaussian());
    cell->SetNHasGaussian((*it)->GetNHasGaussian());
    cell->SetSkewRad(skew_rad);
    cell->SetCenter(R * (*it)->GetCenter() + t);
    cell->SetSize((*it)->GetSize());
    cell->SetPointMean(R * (*it)->GetPointMean() + t);
    if ((*it)->GetPHasGaussian()) {
      cell->SetPointCov(R * (*it)->GetPointCov() * R.transpose());
      cell->SetPointEvals((*it)->GetPointEvals());
      cell->SetPointEvecs(R * (*it)->GetPointEvecs());
    }
    cell->SetNormalMean(R * (*it)->GetNormalMean());
    if ((*it)->GetNHasGaussian()) {
      cell->SetNormalCov(R * (*it)->GetNormalCov() * R.transpose());
      cell->SetNormalEvals((*it)->GetNormalEvals());
      cell->SetNormalEvecs(R * (*it)->GetNormalEvecs());
    }
    if (include_data) {
      for (auto pt : (*it)->GetPoints()) {
        if (pt.allFinite()) pt = R * pt + t;
        cell->AddPoint(pt);
      }
      for (auto nm : (*it)->GetNormals()) {
        if (nm.allFinite()) nm = R * nm;
        cell->AddNormal(nm);
      }
    }
    ret.push_back(cell);
  }
  return ret;
}

SNDTCell *SNDTMap::GetCellAndAllocate(const Eigen::Vector2d &point) {
  if (!point.allFinite()) {
    return nullptr;
  }
  auto idx = GetIndexForPoint(point);
  if (!IsValidIndex(idx)) {
    return nullptr;
  }
  if (!cellptrs_[idx(0)][idx(1)]) {
    cellptrs_[idx(0)][idx(1)] = new SNDTCell();
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

const SNDTCell *SNDTMap::GetCellForPoint(const Eigen::Vector2d &point) const {
  if (!is_loaded_) {
    return nullptr;
  }
  auto idx = GetIndexForPoint(point);
  if (!IsValidIndex(idx)) {
    return nullptr;
  }
  return cellptrs_[idx(0)][idx(1)];
}

std::string SNDTMap::ToString() const {
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
    char s[20];
    sprintf(s, "[#%ld, (%d, %d)] ", i, idx(0), idx(1));
    ret += std::string(s) + cells_[i]->ToString();
  }
  return ret;
}

void SNDTMap::ShowCellDistri() const {
  int one = 0, two = 0, gau = 0;
  int pnoinit = 0, pnopts = 0, pvalid = 0, prescale = 0, passign = 0,
      pinvalid = 0;
  int nnoinit = 0, nnopts = 0, nvalid = 0, nrescale = 0, nassign = 0,
      ninvalid = 0;
  for (auto cell : cells_) {
    if (cell->GetPCellType() == SNDTCell::kNoInit) ++pnoinit;
    if (cell->GetNCellType() == SNDTCell::kNoInit) ++nnoinit;
    if (cell->GetPCellType() == SNDTCell::kNoPoints) ++pnopts;
    if (cell->GetNCellType() == SNDTCell::kNoPoints) ++nnopts;
    if (cell->GetPCellType() == SNDTCell::kRegular) ++pvalid;
    if (cell->GetNCellType() == SNDTCell::kRegular) ++nvalid;
    if (cell->GetPCellType() == SNDTCell::kRescale) ++prescale;
    if (cell->GetNCellType() == SNDTCell::kRescale) ++nrescale;
    if (cell->GetPCellType() == SNDTCell::kAssign) ++passign;
    if (cell->GetNCellType() == SNDTCell::kAssign) ++nassign;
    if (cell->GetPCellType() == SNDTCell::kInvalid) ++pinvalid;
    if (cell->GetNCellType() == SNDTCell::kInvalid) ++ninvalid;
    if (cell->GetN() == 1) ++one;
    if (cell->GetN() == 2) ++two;
    if (cell->HasGaussian()) ++gau;
  }
  ::printf("%ld cells: %d one, %d two, %d gau\n", cells_.size(), one, two, gau);
  ::printf("p: nin: %d, npt: %d, reg: %d, res: %d, ass: %d, inv: %d\n", pnoinit,
           pnopts, pvalid, prescale, passign, pinvalid);
  ::printf("n: nin: %d, npt: %d, reg: %d, res: %d, ass: %d, inv: %d\n\n",
           nnoinit, nnopts, nvalid, nrescale, nassign, ninvalid);
}

std::vector<Eigen::Vector2d> SNDTMap::GetPoints() const {
  std::vector<Eigen::Vector2d> ret;
  for (auto it = begin(); it != end(); ++it)
    for (auto pt : (*it)->GetPoints()) ret.push_back(pt);
  return ret;
}

std::vector<Eigen::Vector2d> SNDTMap::GetPointsWithGaussianCell() const {
  std::vector<Eigen::Vector2d> ret;
  for (auto it = begin(); it != end(); ++it)
    if ((*it)->HasGaussian())
      for (auto pt : (*it)->GetPoints()) ret.push_back(pt);
  return ret;
}

std::vector<Eigen::Vector2d> SNDTMap::GetNormals() const {
  std::vector<Eigen::Vector2d> ret;
  for (auto it = begin(); it != end(); ++it)
    for (auto nm : (*it)->GetNormals()) ret.push_back(nm);
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
int SNDTMap::GetCellIndex(const SNDTCell *cell) const {
  auto c = const_cast<SNDTCell *>(cell);
  if (!cell_idx_map_.count(c)) {
    std::cerr << __FUNCTION__ << ": not find query cell\n";
    std::exit(1);
  }
  return cell_idx_map_.at(c);
}

void SNDTMap::Initialize() {
  cellptrs_ = new SNDTCell **[cellptrs_size_(0)];
  for (int i = 0; i < cellptrs_size_(0); ++i) {
    cellptrs_[i] = new SNDTCell *[cellptrs_size_(1)];
    memset(cellptrs_[i], 0, cellptrs_size_(1) * sizeof(SNDTCell *));
  }
}
