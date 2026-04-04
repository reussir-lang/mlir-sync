// RUN: %not %sync-opt %s -verify-diagnostics

func.func @rank_must_be_zero(%once: memref<1x!sync.once>) {
  sync.once.init %once : memref<1x!sync.once>
  // expected-error @above {{once must be a zero-ranked memref}}
  return
}

func.func @element_must_match(%once: memref<i32>) {
  %completed = sync.once.completed %once : memref<i32>
  // expected-error @above {{once element type must be !sync.once}}
  return
}

func.func @execute_body_takes_no_arguments(%once: memref<!sync.once>) {
  sync.once.execute %once : memref<!sync.once> {
  ^bb0(%arg0: i32):
    sync.yield
  }
  // expected-error @above {{body must not take block arguments}}
  return
}

func.func @execute_body_must_not_yield_values(%once: memref<!sync.once>) {
  %c1 = arith.constant 1 : i32
  sync.once.execute %once : memref<!sync.once> {
    sync.yield %c1 : i32
  }
  // expected-error @above {{body must not yield values}}
  return
}
