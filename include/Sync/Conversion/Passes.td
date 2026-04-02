#ifndef SYNC_CONVERSION_PASSES_TD
#define SYNC_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertSyncToSTDPass : Pass<"convert-sync-to-std", "::mlir::ModuleOp"> {
  let summary = "separate sync control flow into scf and bridge operations";
  let description = [{
    `convert-sync-to-std` rewrites sync operations that introduce control-flow
    structure into `scf` and `func.call`, leaving the non-control-flow fast
    path operations for the final lowering pass.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::ptr::PtrDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::sync::SyncDialect"
  ];
}

def ConvertSyncToLLVMPass : Pass<"convert-sync-to-llvm", "::mlir::ModuleOp"> {
  let summary = "lower sync bridge operations and types to LLVM";
  let description = [{
    `convert-sync-to-llvm` lowers non-control-flow sync bridge operations to
    the LLVM dialect and converts sync types through the LLVM type converter.
  }];
  let dependentDialects = [
    "::mlir::cf::ControlFlowDialect",
    "::mlir::func::FuncDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::ptr::PtrDialect",
    "::mlir::sync::SyncDialect"
  ];
}

#endif // SYNC_CONVERSION_PASSES_TD
