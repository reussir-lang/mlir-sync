// RUN: %sync-opt %s --attach-host-llvm-layout --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm > %t
// RUN: %mlir-translate --mlir-to-llvmir %t | %clang -Wno-override-module -O0 -c -x ir -o %t.o -
// RUN: %clang -std=c11 -O0 -fopenmp %S/mutex_i64_omp_e2e_main.c %t.o %sync-runtime-lib -o %t.exe
// RUN: %t.exe

module {
  func.func @typed_mutex_parallel_increment() -> i64 {
    %num_threads = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %iters = arith.constant 1000 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    %mutex = memref.alloca() : memref<!sync.mutex<i64>>
    sync.mutex.init %mutex : memref<!sync.mutex<i64>>, %c0_i64 : i64

    omp.parallel num_threads(%num_threads : i32) {
      scf.for %i = %c0 to %iters step %c1 {
        sync.mutex.critical_section %mutex : memref<!sync.mutex<i64>> {
        ^bb0(%payload: memref<i64>):
          %value = memref.load %payload[] : memref<i64>
          %next = arith.addi %value, %c1_i64 : i64
          memref.store %next, %payload[] : memref<i64>
          sync.yield
        }
      }
      omp.terminator
    }

    %result = sync.mutex.critical_section %mutex
      : memref<!sync.mutex<i64>> -> i64 {
    ^bb0(%payload: memref<i64>):
      %value = memref.load %payload[] : memref<i64>
      sync.yield %value : i64
    }
    return %result : i64
  }
}
