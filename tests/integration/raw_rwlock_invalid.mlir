// RUN: %not %sync-opt %s -verify-diagnostics

func.func @rank_must_be_zero(%rwlock: memref<1x!sync.raw_rwlock>) {
  sync.raw_rwlock.read_lock %rwlock : memref<1x!sync.raw_rwlock>
  // expected-error @above {{rwlock must be a zero-ranked memref}}
  return
}

func.func @element_must_match(%rwlock: memref<i32>) {
  sync.raw_rwlock.try_write_lock %rwlock : memref<i32>
  // expected-error @above {{rwlock element type must be !sync.raw_rwlock}}
  return
}
