#include "Sync/IR/SyncOps.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

#define GET_OP_CLASSES
#include "Sync/IR/SyncOps.cpp.inc"

namespace mlir::sync {
namespace {

mlir::LogicalResult verifyRawMutexMemRef(mlir::Operation *op, mlir::Type type,
                                         llvm::StringRef operandName) {
  auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return op->emitOpError() << operandName << " must be a ranked memref";
  if (memrefType.getRank() != 0)
    return op->emitOpError() << operandName
                             << " must be a zero-ranked memref, got "
                             << memrefType;
  if (!llvm::isa<RawMutexType>(memrefType.getElementType()))
    return op->emitOpError() << operandName
                             << " element type must be !sync.raw_mutex, got "
                             << memrefType.getElementType();
  return mlir::success();
}

} // namespace

mlir::LogicalResult SyncRawMutexInitOp::verify() {
  return verifyRawMutexMemRef(*this, getMutex().getType(), "mutex");
}

mlir::LogicalResult SyncRawMutexTryLockOp::verify() {
  return verifyRawMutexMemRef(*this, getMutex().getType(), "mutex");
}

mlir::LogicalResult SyncRawMutexLockOp::verify() {
  return verifyRawMutexMemRef(*this, getMutex().getType(), "mutex");
}

mlir::LogicalResult SyncRawMutexUnlockFastOp::verify() {
  return verifyRawMutexMemRef(*this, getMutex().getType(), "mutex");
}

mlir::LogicalResult SyncRawMutexUnlockOp::verify() {
  return verifyRawMutexMemRef(*this, getMutex().getType(), "mutex");
}

} // namespace mlir::sync
