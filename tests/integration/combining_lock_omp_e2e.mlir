// RUN: %sync-opt %s --attach-host-llvm-layout --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm > %t
// RUN: %mlir-translate --mlir-to-llvmir %t | %clang -Wno-override-module -O0 -c -x ir -o %t.o -
// RUN: %clang -std=c11 -O0 -fopenmp %S/combining_lock_omp_e2e_main.c %t.o %sync-runtime-lib -o %t.exe
// RUN: %t.exe

module {
  func.func @combining_lock_parallel_increment() -> i64 {
    %num_threads = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %iters = arith.constant 1000 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %lock = memref.alloca() : memref<!sync.combining_lock<i64>>
    sync.combining_lock.init %lock : memref<!sync.combining_lock<i64>>, %c0_i64 : i64

    omp.parallel num_threads(%num_threads : i32) {
      scf.for %i = %c0 to %iters step %c1 {
        sync.combining_lock.critical_section %lock : memref<!sync.combining_lock<i64>> {
        ^bb0(%payload: memref<i64>):
          %value = memref.load %payload[] : memref<i64>
          %next = arith.addi %value, %c1_i64 : i64
          memref.store %next, %payload[] : memref<i64>
          sync.yield
        }
      }
      omp.terminator
    }

    %payload = sync.combining_lock.get_payload %lock : memref<!sync.combining_lock<i64>> -> memref<i64>
    %result = memref.load %payload[] : memref<i64>
    return %result : i64
  }
}
