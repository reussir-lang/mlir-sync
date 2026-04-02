// RUN: %not %sync-opt %s -verify-diagnostics

func.func @rank_must_be_zero(%mutex: memref<1x!sync.raw_mutex>) {
  sync.raw_mutex.lock %mutex : memref<1x!sync.raw_mutex>
  // expected-error @above {{mutex must be a zero-ranked memref}}
  return
}

func.func @element_must_match(%mutex: memref<i32>) {
  sync.raw_mutex.try_lock %mutex : memref<i32>
  // expected-error @above {{mutex element type must be !sync.raw_mutex}}
  return
}
