// RUN: %sync-opt %s --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm > %t
// RUN: %mlir-translate --mlir-to-llvmir %t | %clang -Wno-override-module -O0 -c -x ir -o %t.o -
// RUN: %clang -std=c11 -O0 -fopenmp %S/raw_rwlock_omp_e2e_main.c %t.o %sync-runtime-lib -o %t.exe
// RUN: %t.exe

module {
  func.func private @omp_get_thread_num() -> i32

  func.func @raw_rwlock_parallel_read_sum() -> i64 {
    %num_threads_i32 = arith.constant 4 : i32
    %num_threads = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %iters = arith.constant 1000 : index
    %c0_i64 = arith.constant 0 : i64
    %c7_i64 = arith.constant 7 : i64

    %rwlock = memref.alloca() : memref<!sync.raw_rwlock>
    %value = memref.alloca() : memref<i64>
    %partials = memref.alloca() : memref<4xi64>
    sync.raw_rwlock.init %rwlock : memref<!sync.raw_rwlock>
    memref.store %c7_i64, %value[] : memref<i64>

    scf.for %i = %c0 to %num_threads step %c1 {
      memref.store %c0_i64, %partials[%i] : memref<4xi64>
    }

    omp.parallel num_threads(%num_threads_i32 : i32) {
      %tid_i32 = func.call @omp_get_thread_num() : () -> i32
      %tid = arith.index_cast %tid_i32 : i32 to index
      %local_sum = scf.for %i = %c0 to %iters step %c1 iter_args(%acc = %c0_i64) -> (i64) {
        sync.raw_rwlock.read_lock %rwlock : memref<!sync.raw_rwlock>
        %loaded = memref.load %value[] : memref<i64>
        sync.raw_rwlock.read_unlock %rwlock : memref<!sync.raw_rwlock>
        %next = arith.addi %acc, %loaded : i64
        scf.yield %next : i64
      }
      memref.store %local_sum, %partials[%tid] : memref<4xi64>
      omp.terminator
    }

    %result = scf.for %i = %c0 to %num_threads step %c1 iter_args(%acc = %c0_i64) -> (i64) {
      %partial = memref.load %partials[%i] : memref<4xi64>
      %next = arith.addi %acc, %partial : i64
      scf.yield %next : i64
    }
    return %result : i64
  }
}
