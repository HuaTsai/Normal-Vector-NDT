#pragma once

enum class Options {
  kNDT,
  kNormalNDT,
  kSICP,
  k1to1,
  k1ton,
  kIterative,
  kPointCov,
  kNoReject,
  kOptimizer2D,
  kOptimizer3D,
  kLBFGSPP,
  kAnalytic,
  kUseNormalOMP
};

constexpr Options kNDT = Options::kNDT;
constexpr Options kNNDT = Options::kNormalNDT;
constexpr Options kSICP = Options::kSICP;
constexpr Options k1to1 = Options::k1to1;
constexpr Options k1ton = Options::k1ton;
constexpr Options kIterative = Options::kIterative;
constexpr Options kPointCov = Options::kPointCov;
constexpr Options kNoReject = Options::kNoReject;
constexpr Options kOptimizer2D = Options::kOptimizer2D;
constexpr Options kOptimizer3D = Options::kOptimizer3D;
constexpr Options kLBFGSPP = Options::kLBFGSPP;
constexpr Options kAnalytic = Options::kAnalytic;
constexpr Options kUseNormalOMP = Options::kUseNormalOMP;
