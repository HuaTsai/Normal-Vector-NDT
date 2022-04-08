#pragma once

enum class Options {
  kLineSearch,
  kTrustRegion,
  kNDT,
  kNormalNDT,
  k1to1,
  k1ton,
  kIterative,
  kPointCov,
  kNoReject
};

constexpr Options kLS = Options::kLineSearch;
constexpr Options kTR = Options::kTrustRegion;
constexpr Options kNDT = Options::kNDT;
constexpr Options kNNDT = Options::kNormalNDT;
constexpr Options k1to1 = Options::k1to1;
constexpr Options k1ton = Options::k1ton;
constexpr Options kIterative = Options::kIterative;
constexpr Options kPointCov = Options::kPointCov;
constexpr Options kNoReject = Options::kNoReject;
