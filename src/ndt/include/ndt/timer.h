#pragma once
#include <bits/stdc++.h>

class Timer {
  using TP = std::chrono::steady_clock::time_point;
  enum State { kInit, kStart, kSubStart, kSubFinish, kFinish };

 public:
  enum Procedure { kNormal, kNDT, kBuild, kOptimize };
  Timer();

  Timer &operator+=(const Timer &rhs);
  const Timer operator+(const Timer &rhs);

  void Start();
  void ProcedureStart(Procedure procedure);
  void ProcedureFinish();
  void Finish();
  void Show();

  int normal() const { return data_.at(kNormal); }
  int ndt() const { return data_.at(kNDT); }
  int build() const { return data_.at(kBuild); }
  int optimize() const { return data_.at(kOptimize); }
  int others() const { return others_; }
  int total() const { return total_; }

 private:
  State state_;
  Procedure current_procedure_;
  TP start_, end_, current_start_, current_end_;
  std::unordered_map<Procedure, int> data_;
  int others_;
  int total_;
};
