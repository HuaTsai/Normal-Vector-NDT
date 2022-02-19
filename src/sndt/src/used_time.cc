#include <common/other_utils.h>
#include <sndt/used_time.h>

void UsedTime::Start() {
  if (!(state_ == kInit)) {
    std::cerr << __FUNCTION__ << ": invalid state " << state_ << std::endl;
    return;
  }
  start_ = GetTime();
  state_ = kStart;
}

void UsedTime::ProcedureStart(Procedure procedure) {
  if (!(state_ == kStart || state_ == kSubFinish)) {
    std::cerr << __FUNCTION__ << ": invalid state " << state_ << std::endl;
    return;
  }
  current_start_ = GetTime();
  current_procedure_ = procedure;
  state_ = kSubStart;
}

void UsedTime::ProcedureFinish() {
  if (!(state_ == kSubStart)) {
    std::cerr << __FUNCTION__ << ": invalid state " << state_ << std::endl;
    return;
  }
  current_end_ = GetTime();
  int ms = GetDiffTime(current_start_, current_end_);
  if (current_procedure_ == Procedure::kNormal) {
    normal_ += ms;
  } else if (current_procedure_ == Procedure::kNDT) {
    ndt_ += ms;
  } else if (current_procedure_ == Procedure::kBuild) {
    build_ += ms;
  } else if (current_procedure_ == Procedure::kOptimize) {
    optimize_ += ms;
  }
  state_ = kSubFinish;
}

void UsedTime::Finish() {
  if (!(state_ == kStart || state_ == kSubFinish)) {
    std::cerr << __FUNCTION__ << ": invalid state " << state_ << std::endl;
    return;
  }
  end_ = GetTime();
  total_ = GetDiffTime(start_, end_);
  others_ = total_ - normal_ - ndt_ - build_ - optimize_;
  state_ = kFinish;
}
