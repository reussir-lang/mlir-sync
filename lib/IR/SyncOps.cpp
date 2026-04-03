#include "Sync/IR/SyncOps.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
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

mlir::LogicalResult verifyCombiningLockMemRef(mlir::Operation *op,
                                              mlir::Type type,
                                              llvm::StringRef operandName) {
  auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return op->emitOpError() << operandName << " must be a ranked memref";
  if (memrefType.getRank() != 0)
    return op->emitOpError() << operandName
                             << " must be a zero-ranked memref, got "
                             << memrefType;
  if (!llvm::isa<CombiningLockType>(memrefType.getElementType()))
    return op->emitOpError()
           << operandName
           << " element type must be !sync.combining_lock<...>, got "
           << memrefType.getElementType();
  return mlir::success();
}

mlir::LogicalResult verifyCombiningLockNodeMemRef(mlir::Operation *op,
                                                  mlir::Type type,
                                                  llvm::StringRef operandName) {
  auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return op->emitOpError() << operandName << " must be a ranked memref";
  if (memrefType.getRank() != 0)
    return op->emitOpError() << operandName
                             << " must be a zero-ranked memref, got "
                             << memrefType;
  if (!llvm::isa<CombiningLockNodeType>(memrefType.getElementType()))
    return op->emitOpError()
           << operandName
           << " element type must be !sync.combining_lock_node<...>, got "
           << memrefType.getElementType();
  return mlir::success();
}

mlir::LogicalResult verifyPtrType(mlir::Operation *op, mlir::Type type,
                                  llvm::StringRef operandName) {
  if (!llvm::isa<mlir::ptr::PtrType>(type))
    return op->emitOpError() << operandName << " must have type !ptr.ptr<...>, got "
                             << type;
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

mlir::MemRefType getCombiningLockPayloadProjectionType(
    mlir::MemRefType lockType) {
  auto lockElementType =
      llvm::cast<CombiningLockType>(lockType.getElementType());
  return mlir::MemRefType::get({}, lockElementType.getValueType(),
                               mlir::MemRefLayoutAttrInterface{},
                               lockType.getMemorySpace());
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

mlir::LogicalResult SyncCombiningLockInitOp::verify() {
  if (mlir::failed(verifyCombiningLockMemRef(*this, getLock().getType(), "lock")))
    return mlir::failure();

  if (mlir::Value initialValue = getInitialValue()) {
    auto lockType = llvm::cast<mlir::MemRefType>(getLock().getType());
    auto payloadType =
        getCombiningLockPayloadProjectionType(lockType).getElementType();
    if (initialValue.getType() != payloadType)
      return emitOpError() << "initial value type must be " << payloadType
                           << ", got " << initialValue.getType();
  }
  return mlir::success();
}

mlir::LogicalResult SyncCombiningLockHasTailOp::verify() {
  return verifyCombiningLockMemRef(*this, getLock().getType(), "lock");
}

mlir::LogicalResult SyncCombiningLockTryAcquireOp::verify() {
  return verifyCombiningLockMemRef(*this, getLock().getType(), "lock");
}

mlir::LogicalResult SyncCombiningLockReleaseOp::verify() {
  return verifyCombiningLockMemRef(*this, getLock().getType(), "lock");
}

mlir::LogicalResult SyncCombiningLockGetPayloadOp::verify() {
  auto sourceType = llvm::dyn_cast<mlir::MemRefType>(getSource().getType());
  if (!sourceType)
    return emitOpError() << "source must be a ranked memref";
  if (sourceType.getRank() != 0)
    return emitOpError() << "source must be a zero-ranked memref, got "
                         << sourceType;

  if (auto lockType =
          llvm::dyn_cast<CombiningLockType>(sourceType.getElementType())) {
    auto expectedType = mlir::MemRefType::get({}, lockType.getValueType(),
                                              mlir::MemRefLayoutAttrInterface{},
                                              sourceType.getMemorySpace());
    if (getNumResults() != 1)
      return emitOpError() << "combining lock projection must return one result";
    if (getResult(0).getType() != expectedType)
      return emitOpError() << "result type must be " << expectedType << ", got "
                           << getResult(0).getType();
    return mlir::success();
  }

  auto nodeType =
      llvm::dyn_cast<CombiningLockNodeType>(sourceType.getElementType());
  if (!nodeType)
    return emitOpError()
           << "source element type must be !sync.combining_lock<...> or "
              "!sync.combining_lock_node<...>, got "
           << sourceType.getElementType();
  if (getNumResults() != nodeType.getCaptureTypes().size() + 1)
    return emitOpError() << "node projection must return one payload memref and "
                         << nodeType.getCaptureTypes().size()
                         << " captures, got " << getNumResults() << " results";

  auto payloadType = llvm::dyn_cast<mlir::MemRefType>(getResult(0).getType());
  if (!payloadType || payloadType.getRank() != 0)
    return emitOpError()
           << "first result must be a zero-ranked payload memref, got "
           << getResult(0).getType();

  for (auto [index, captureType] : llvm::enumerate(nodeType.getCaptureTypes())) {
    auto resultType = getResult(static_cast<unsigned>(index) + 1).getType();
    if (resultType != captureType)
      return emitOpError() << "capture result type must be " << captureType
                           << ", got " << resultType;
  }
  return mlir::success();
}

mlir::LogicalResult SyncCombiningLockCaptureOp::verify() {
  auto calleeType = llvm::dyn_cast<mlir::FunctionType>(getCallee().getType());
  if (!calleeType)
    return emitOpError() << "callee must have a function type";
  if (calleeType.getNumInputs() != 1 || calleeType.getNumResults() != 0)
    return emitOpError() << "callee must have type (!ptr.ptr<...>) -> ()";
  if (!llvm::isa<mlir::ptr::PtrType>(calleeType.getInput(0)))
    return emitOpError() << "callee input must be !ptr.ptr<...>, got "
                         << calleeType.getInput(0);

  auto payloadType = llvm::dyn_cast<mlir::MemRefType>(getPayload().getType());
  if (!payloadType)
    return emitOpError() << "payload must be a ranked memref";
  if (payloadType.getRank() != 0)
    return emitOpError() << "payload must be a zero-ranked memref, got "
                         << payloadType;

  if (mlir::failed(verifyCombiningLockNodeMemRef(*this, getNode().getType(), "node")))
    return mlir::failure();
  if (mlir::failed(verifyPtrType(*this, getRawNode().getType(), "raw node")))
    return mlir::failure();

  auto nodeType = llvm::cast<mlir::MemRefType>(getNode().getType());
  auto captures =
      llvm::cast<CombiningLockNodeType>(nodeType.getElementType()).getCaptureTypes();
  if (captures.size() != getCaptures().size())
    return emitOpError() << "node type expects " << captures.size()
                         << " captures, got " << getCaptures().size();
  for (auto [captureValue, captureType] :
       llvm::zip_equal(getCaptures(), captures)) {
    if (captureValue.getType() != captureType)
      return emitOpError() << "capture type must be " << captureType
                           << ", got " << captureValue.getType();
  }
  return mlir::success();
}

mlir::LogicalResult SyncCombiningLockRecoverOp::verify() {
  if (mlir::failed(verifyPtrType(*this, getNode().getType(), "node")))
    return mlir::failure();
  if (getNumResults() == 0)
    return emitOpError() << "must return at least the payload memref";
  auto payloadType = llvm::dyn_cast<mlir::MemRefType>(getResult(0).getType());
  if (!payloadType || payloadType.getRank() != 0)
    return emitOpError()
           << "first result must be a zero-ranked payload memref, got "
           << getResult(0).getType();
  return mlir::success();
}

mlir::LogicalResult SyncCombiningLockCaptureEndOp::verify() {
  return verifyPtrType(*this, getRawNode().getType(), "raw node");
}

mlir::LogicalResult SyncCombiningLockCriticalSectionOp::verify() {
  if (mlir::failed(verifyCombiningLockMemRef(*this, getLock().getType(), "lock")))
    return mlir::failure();
  if (auto combineLimit = getCombineLimitAttr();
      combineLimit && combineLimit.getInt() <= 0)
    return emitOpError() << "combine_limit must be positive";
  return mlir::success();
}

mlir::LogicalResult SyncCombiningLockCriticalSectionOp::verifyRegions() {
  auto &block = getBody().front();
  if (block.getNumArguments() != 1)
    return emitOpError() << "body must have exactly one block argument";

  auto lockType = llvm::cast<mlir::MemRefType>(getLock().getType());
  auto expectedPayloadType = getCombiningLockPayloadProjectionType(lockType);
  if (block.getArgument(0).getType() != expectedPayloadType)
    return emitOpError() << "body argument must have type "
                         << expectedPayloadType << ", got "
                         << block.getArgument(0).getType();

  auto yieldOp = llvm::dyn_cast<SyncYieldOp>(block.getTerminator());
  if (!yieldOp)
    return emitOpError() << "body must terminate with sync.yield";
  if (yieldOp->getNumOperands() != 0)
    return emitOpError()
           << "combining lock critical sections cannot yield values";
  return mlir::success();
}

} // namespace mlir::sync
