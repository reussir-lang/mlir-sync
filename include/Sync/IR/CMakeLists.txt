set(LLVM_TARGET_DEFINITIONS SyncDialect.td)
mlir_tablegen(SyncDialect.h.inc -gen-dialect-decls -dialect=sync)
mlir_tablegen(SyncDialect.cpp.inc -gen-dialect-defs -dialect=sync)

set(LLVM_TARGET_DEFINITIONS SyncTypes.td)
mlir_tablegen(SyncTypes.h.inc -gen-typedef-decls)
mlir_tablegen(SyncTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS SyncOps.td)
mlir_tablegen(SyncOps.h.inc -gen-op-decls)
mlir_tablegen(SyncOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(SyncOpsIncGen)
