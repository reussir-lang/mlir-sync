#include "Sync/IR/SyncDialect.h"

#include "Sync/IR/SyncDialect.cpp.inc"
#include "Sync/IR/SyncOps.h"
#include "Sync/IR/SyncTypes.h"

namespace mlir::sync {

void SyncDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Sync/IR/SyncTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Sync/IR/SyncOps.cpp.inc"
      >();
}

} // namespace mlir::sync
