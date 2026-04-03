#include "Sync/IR/SyncDialect.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

#include "Sync/IR/SyncOps.h"
#include "Sync/IR/SyncTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Sync/IR/SyncTypes.cpp.inc"

#include "Sync/IR/SyncDialect.cpp.inc"

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
