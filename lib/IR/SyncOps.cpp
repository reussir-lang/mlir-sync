#include "Sync/IR/SyncOps.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>

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

mlir::LogicalResult verifyMutexMemRef(mlir::Operation *op, mlir::Type type,
                                      llvm::StringRef operandName) {
  auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return op->emitOpError() << operandName << " must be a ranked memref";
  if (memrefType.getRank() != 0)
    return op->emitOpError() << operandName
                             << " must be a zero-ranked memref, got "
                             << memrefType;
  if (!llvm::isa<MutexType>(memrefType.getElementType()))
    return op->emitOpError() << operandName
                             << " element type must be !sync.mutex<...>, got "
                             << memrefType.getElementType();
  return mlir::success();
}

mlir::MemRefType getRawMutexProjectionType(mlir::MemRefType mutexType) {
  return mlir::MemRefType::get(
      {}, RawMutexType::get(mutexType.getContext()),
      mlir::MemRefLayoutAttrInterface{}, mutexType.getMemorySpace());
}

mlir::MemRefType getPayloadProjectionType(mlir::MemRefType mutexType) {
  auto mutexElementType = llvm::cast<MutexType>(mutexType.getElementType());
  return mlir::MemRefType::get({}, mutexElementType.getValueType(),
                               mlir::MemRefLayoutAttrInterface{},
                               mutexType.getMemorySpace());
}

} // namespace

mlir::LogicalResult SyncRawMutexInitOp::verify() {
  return verifyRawMutexMemRef(*this, getMutex().getType(), "mutex");
}

mlir::LogicalResult SyncMutexInitOp::verify() {
  if (mlir::failed(verifyMutexMemRef(*this, getMutex().getType(), "mutex")))
    return mlir::failure();

  if (mlir::Value initialValue = getInitialValue()) {
    auto mutexType = llvm::cast<mlir::MemRefType>(getMutex().getType());
    auto payloadType = getPayloadProjectionType(mutexType).getElementType();
    if (initialValue.getType() != payloadType)
      return emitOpError() << "initial value type must be " << payloadType
                           << ", got " << initialValue.getType();
  }
  return mlir::success();
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

mlir::LogicalResult SyncMutexGetRawMutexOp::verify() {
  if (mlir::failed(verifyMutexMemRef(*this, getMutex().getType(), "mutex")))
    return mlir::failure();

  auto mutexType = llvm::cast<mlir::MemRefType>(getMutex().getType());
  auto expectedType = getRawMutexProjectionType(mutexType);
  if (getRawMutex().getType() != expectedType)
    return emitOpError() << "result type must be " << expectedType << ", got "
                         << getRawMutex().getType();
  return mlir::success();
}

mlir::LogicalResult SyncMutexGetPayloadOp::verify() {
  if (mlir::failed(verifyMutexMemRef(*this, getMutex().getType(), "mutex")))
    return mlir::failure();

  auto mutexType = llvm::cast<mlir::MemRefType>(getMutex().getType());
  auto expectedType = getPayloadProjectionType(mutexType);
  if (getPayload().getType() != expectedType)
    return emitOpError() << "result type must be " << expectedType << ", got "
                         << getPayload().getType();
  return mlir::success();
}

mlir::LogicalResult SyncMutexCriticalSectionOp::verify() {
  return verifyMutexMemRef(*this, getMutex().getType(), "mutex");
}

mlir::LogicalResult SyncMutexCriticalSectionOp::verifyRegions() {
  auto &block = getBody().front();
  if (block.getNumArguments() != 1)
    return emitOpError() << "body must have exactly one block argument";

  auto mutexType = llvm::cast<mlir::MemRefType>(getMutex().getType());
  auto expectedPayloadType = getPayloadProjectionType(mutexType);
  if (block.getArgument(0).getType() != expectedPayloadType)
    return emitOpError() << "body argument must have type "
                         << expectedPayloadType << ", got "
                         << block.getArgument(0).getType();

  auto yieldOp = llvm::dyn_cast<SyncYieldOp>(block.getTerminator());
  if (!yieldOp)
    return emitOpError() << "body must terminate with sync.yield";
  if (yieldOp->getNumOperands() != getNumResults())
    return emitOpError() << "body must yield " << getNumResults()
                         << " values, got " << yieldOp->getNumOperands();
  for (auto [yieldedType, resultType] :
       llvm::zip_equal(yieldOp->getOperandTypes(), getResultTypes())) {
    if (yieldedType != resultType)
      return emitOpError() << "body yielded type " << yieldedType
                           << " but operation result type is " << resultType;
  }
  return mlir::success();
}

} // namespace mlir::sync
