#pragma once
#include <bits/stdc++.h>

class UsedTime {
  typedef std::chrono::steady_clock::time_point TP;
  enum State { kInit, kStart, kSubStart, kSubFinish, kFinish };

 public:
  enum Procedure { kNormal, kNDT, kBuild, kOptimize };
  UsedTime()
      : state_(kInit),
        normal_(0),
        ndt_(0),
        build_(0),
        optimize_(0),
        others_(0),
        total_(0) {}

  UsedTime &operator+=(const UsedTime &rhs) {
    normal_ += rhs.normal();
    ndt_ += rhs.ndt();
    build_ += rhs.build();
    optimize_ += rhs.optimize();
    others_ += rhs.others();
    total_ += rhs.total();
    return *this;
  }

  const UsedTime operator+(const UsedTime &rhs) {
    return UsedTime(*this) += rhs;
  }

  void Start();
  void ProcedureStart(Procedure procedure);
  void ProcedureFinish();
  void Finish();
  void Show() {
    std::cout << "nm: " << normal_ << ", ndt: " << ndt_ << ", bud: " << build_
              << ", opt: " << optimize_ << ", oth: " << others_
              << ", ttl: " << total_ << std::endl;
  }

  int normal() const { return normal_; }
  int ndt() const { return ndt_; }
  int build() const { return build_; }
  int optimize() const { return optimize_; }
  int others() const { return others_; }
  int total() const { return total_; }

 private:
  State state_;
  Procedure current_procedure_;
  TP start_, end_, current_start_, current_end_;
  int normal_;
  int ndt_;
  int build_;
  int optimize_;
  int others_;
  int total_;
};
