#include <gtest/gtest.h>
#include <ndt/timer.h>

TEST(TimerTest, Basic) {
  Timer timer;
  timer.Start();
  sleep(1);
  timer.ProcedureStart(Timer::Procedure::kBuild);
  sleep(2);
  timer.ProcedureFinish();
  timer.ProcedureStart(Timer::Procedure::kOptimize);
  sleep(3);
  timer.ProcedureFinish();
  timer.Finish();
  // Tolerance: 0.1%
  EXPECT_NEAR(timer.build(), 2000000, 2000);
  EXPECT_NEAR(timer.optimize(), 3000000, 3000);
  EXPECT_NEAR(timer.total(), 6000000, 6000);
}

TEST(TimerTest, Addition) {
  Timer timer1;
  timer1.Start();
  timer1.ProcedureStart(Timer::Procedure::kBuild);
  sleep(2);
  timer1.ProcedureFinish();
  timer1.Finish();

  Timer timer2;
  timer2.Start();
  timer2.ProcedureStart(Timer::Procedure::kBuild);
  sleep(2);
  timer2.ProcedureFinish();
  timer2.Finish();

  Timer timer3 = timer1 + timer2;
  // Tolerance: 0.1%
  EXPECT_NEAR(timer3.build(), 4000000, 4000);
}
