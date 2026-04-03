// RUN: %not %sync-opt %s -verify-diagnostics

func.func @raw_projection_type_must_match(%rwlock: memref<!sync.rwlock<i64>>) {
  %raw = sync.rwlock.get_raw_rwlock %rwlock
    : memref<!sync.rwlock<i64>> -> memref<!sync.raw_mutex>
  // expected-error @above {{result type must be memref<!sync.raw_rwlock>}}
  return
}

func.func @payload_projection_type_must_match(%rwlock: memref<!sync.rwlock<i64>>) {
  %payload = sync.rwlock.get_payload %rwlock
    : memref<!sync.rwlock<i64>> -> memref<i32>
  // expected-error @above {{result type must be memref<i64>}}
  return
}

func.func @read_body_argument_must_match(%rwlock: memref<!sync.rwlock<i64>>) {
  sync.rwlock.read_critical_section %rwlock : memref<!sync.rwlock<i64>> {
  ^bb0(%payload: memref<i32>):
    sync.yield
  }
  // expected-error @above {{body argument must have type memref<i64>}}
  return
}
