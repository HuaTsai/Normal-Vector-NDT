#include <common/other_utils.h>
#include <nndt/timer.h>

Timer::Timer() : state_(kInit), others_(0), total_(0) {
  data_[kNormal] = 0;
  data_[kNDT] = 0;
  data_[kBuild] = 0;
  data_[kOptimize] = 0;
}

Timer &Timer::operator+=(const Timer &rhs) {
  for (const auto &elem : rhs.data_) data_[elem.first] += elem.second;
  others_ += rhs.others();
  total_ += rhs.total();
  return *this;
}

const Timer Timer::operator+(const Timer &rhs) { return Timer(*this) += rhs; }

void Timer::Start() {
  if (!(state_ == kInit)) {
    std::cerr << __FUNCTION__ << ": invalid state " << state_ << std::endl;
    return;
  }
  start_ = GetTime();
  state_ = kStart;
}

void Timer::ProcedureStart(Procedure procedure) {
  if (!(state_ == kStart || state_ == kSubFinish)) {
    std::cerr << __FUNCTION__ << ": invalid state " << state_ << std::endl;
    return;
  }
  if (!data_.count(procedure)) {
    std::cerr << __FUNCTION__ << ": invalid procedure " << procedure
              << std::endl;
    return;
  }
  current_start_ = GetTime();
  current_procedure_ = procedure;
  state_ = kSubStart;
}

void Timer::ProcedureFinish() {
  if (!(state_ == kSubStart)) {
    std::cerr << __FUNCTION__ << ": invalid state " << state_ << std::endl;
    return;
  }
  current_end_ = GetTime();
  data_[current_procedure_] += GetDiffTime(current_start_, current_end_);
  state_ = kSubFinish;
}

void Timer::Finish() {
  if (!(state_ == kStart || state_ == kSubFinish)) {
    std::cerr << __FUNCTION__ << ": invalid state " << state_ << std::endl;
    return;
  }
  end_ = GetTime();
  total_ = GetDiffTime(start_, end_);
  others_ =
      total_ - data_[kNormal] - data_[kNDT] - data_[kBuild] - data_[kOptimize];
  state_ = kFinish;
}

void Timer::Show() {
  printf("nm: %.2f, ndt: %.2f, bud: %.2f, opt: %.2f, oth: %.2f, ttl: %.2f\n",
         data_[kNormal] / 1000., data_[kNDT] / 1000., data_[kBuild] / 1000.,
         data_[kOptimize] / 1000., others_ / 1000., total_ / 1000.);
}
