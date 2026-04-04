#include "Sync/Conversion/ConvertSyncToLLVM.h"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/PtrToLLVM/PtrToLLVM.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>

#include "Sync/IR/SyncDialect.h"
#include "Sync/IR/SyncOps.h"

namespace mlir::sync {

#define GEN_PASS_DEF_CONVERTSYNCTOLLVMPASS
#include "Sync/Conversion/Passes.h.inc"

namespace {

constexpr uint64_t kUnlockedState = 0;
constexpr uint64_t kLockedState = 1;
constexpr uint64_t kContendedState = 2;
constexpr uint64_t kOnceIncomplete = 0;
constexpr uint64_t kOnceComplete = 3;
constexpr uint64_t kRwLockReadLocked = 1;
constexpr uint64_t kRwLockMask = (1ull << 30) - 1;
constexpr uint64_t kRwLockWriteLocked = kRwLockMask;
constexpr uint64_t kCombiningNodeWaiting = 0;

mlir::Value createI32Constant(mlir::Location loc, uint64_t value,
                              mlir::ConversionPatternRewriter &rewriter) {
  return mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                        rewriter.getI32IntegerAttr(value));
}

mlir::Value createI8Constant(mlir::Location loc, uint64_t value,
                             mlir::ConversionPatternRewriter &rewriter) {
  return mlir::LLVM::ConstantOp::create(rewriter, loc,
                                        mlir::IntegerType::get(rewriter.getContext(), 8),
                                        rewriter.getI8IntegerAttr(value));
}

mlir::Value createI64Constant(mlir::Location loc, uint64_t value,
                              mlir::ConversionPatternRewriter &rewriter) {
  return mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                        rewriter.getI64IntegerAttr(value));
}

mlir::Type getOpaquePtrType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(context);
}

template <typename OpTy>
mlir::Value getRawMutexPointer(OpTy op, typename OpTy::Adaptor adaptor,
                               const mlir::LLVMTypeConverter &converter,
                               mlir::ConversionPatternRewriter &rewriter) {
  auto memrefType = llvm::cast<mlir::MemRefType>(op.getMutex().getType());
  return mlir::LLVM::getStridedElementPtr(rewriter, op.getLoc(), converter,
                                          memrefType, adaptor.getMutex(), {});
}

template <typename OpTy>
mlir::Value getRawRwLockPointer(OpTy op, typename OpTy::Adaptor adaptor,
                                const mlir::LLVMTypeConverter &converter,
                                mlir::ConversionPatternRewriter &rewriter) {
  auto memrefType = llvm::cast<mlir::MemRefType>(op.getRwlock().getType());
  return mlir::LLVM::getStridedElementPtr(rewriter, op.getLoc(), converter,
                                          memrefType, adaptor.getRwlock(), {});
}

template <typename OpTy>
mlir::Value getOncePointer(OpTy op, typename OpTy::Adaptor adaptor,
                           const mlir::LLVMTypeConverter &converter,
                           mlir::ConversionPatternRewriter &rewriter) {
  auto memrefType = llvm::cast<mlir::MemRefType>(op.getOnce().getType());
  return mlir::LLVM::getStridedElementPtr(rewriter, op.getLoc(), converter,
                                          memrefType, adaptor.getOnce(), {});
}

mlir::Value getMutexPointer(mlir::Value mutex, mlir::MemRefType memrefType,
                            const mlir::LLVMTypeConverter &converter,
                            mlir::Location loc,
                            mlir::ConversionPatternRewriter &rewriter) {
  return mlir::LLVM::getStridedElementPtr(rewriter, loc, converter, memrefType,
                                          mutex, {});
}

mlir::Value getMutexFieldPointer(mlir::Location loc, mlir::Value mutexPtr,
                                 mlir::Type mutexElementType, unsigned field,
                                 mlir::ConversionPatternRewriter &rewriter) {
  llvm::SmallVector<mlir::LLVM::GEPArg> indices{0, static_cast<int32_t>(field)};
  return mlir::LLVM::GEPOp::create(rewriter, loc, mutexPtr.getType(),
                                   mutexElementType, mutexPtr, indices)
      .getResult();
}

mlir::Value getRawRwLockStatePointer(mlir::Location loc, mlir::Value rawRwLockPtr,
                                     mlir::Type rawRwLockType,
                                     mlir::ConversionPatternRewriter &rewriter) {
  return getMutexFieldPointer(loc, rawRwLockPtr, rawRwLockType, /*field=*/0,
                              rewriter);
}

mlir::Value getRawRwLockWriterNotifyPointer(
    mlir::Location loc, mlir::Value rawRwLockPtr, mlir::Type rawRwLockType,
    mlir::ConversionPatternRewriter &rewriter) {
  return getMutexFieldPointer(loc, rawRwLockPtr, rawRwLockType, /*field=*/1,
                              rewriter);
}

mlir::Value getCombiningLockFieldPointer(mlir::Location loc, mlir::Value lockPtr,
                                         mlir::Type lockElementType,
                                         llvm::ArrayRef<int32_t> indices,
                                         mlir::ConversionPatternRewriter &rewriter) {
  llvm::SmallVector<mlir::LLVM::GEPArg> gepIndices;
  gepIndices.reserve(indices.size());
  for (int32_t index : indices)
    gepIndices.push_back(index);
  return mlir::LLVM::GEPOp::create(rewriter, loc, lockPtr.getType(),
                                   lockElementType, lockPtr, gepIndices)
      .getResult();
}

template <typename OpTy>
mlir::Value getCombiningLockPointer(OpTy op, mlir::Value lock,
                                    const mlir::LLVMTypeConverter &converter,
                                    mlir::ConversionPatternRewriter &rewriter) {
  auto memrefType = llvm::cast<mlir::MemRefType>(op.getLock().getType());
  return mlir::LLVM::getStridedElementPtr(rewriter, op.getLoc(), converter,
                                          memrefType, lock, {});
}

mlir::Value getNodePointer(mlir::Operation *op, mlir::Value node,
                           mlir::MemRefType nodeType,
                           const mlir::LLVMTypeConverter &converter,
                           mlir::ConversionPatternRewriter &rewriter) {
  return mlir::LLVM::getStridedElementPtr(rewriter, op->getLoc(), converter,
                                          nodeType, node, {});
}

mlir::Value buildStaticZeroRankMemRef(mlir::Location loc,
                                      const mlir::LLVMTypeConverter &converter,
                                      mlir::MemRefType memrefType,
                                      mlir::Value ptr,
                                      mlir::ConversionPatternRewriter &rewriter) {
  auto descriptor = mlir::MemRefDescriptor::fromStaticShape(
      rewriter, loc, converter, memrefType, ptr);
  return mlir::Value(descriptor);
}

mlir::Type getCombiningNodeLLVMTypeForCaptures(
    const mlir::LLVMTypeConverter &converter,
    mlir::MLIRContext *context, mlir::TypeRange captureTypes) {
  llvm::SmallVector<mlir::Type> fields{
      mlir::IntegerType::get(context, 32), getOpaquePtrType(context),
      getOpaquePtrType(context), getOpaquePtrType(context)};
  for (mlir::Type captureType : captureTypes) {
    mlir::Type loweredCapture = converter.convertType(captureType);
    if (!loweredCapture)
      return {};
    fields.push_back(loweredCapture);
  }
  return mlir::LLVM::LLVMStructType::getLiteral(context, fields);
}

struct RawMutexInitLowering
    : public mlir::OpConversionPattern<SyncRawMutexInitOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRawMutexInitOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto ptr = getRawMutexPointer(op, adaptor, converter, rewriter);
    auto zero = createI32Constant(op.getLoc(), kUnlockedState, rewriter);
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), zero, ptr);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RawMutexTryLockLowering
    : public mlir::OpConversionPattern<SyncRawMutexTryLockOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRawMutexTryLockOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto ptr = getRawMutexPointer(op, adaptor, converter, rewriter);
    auto zero = createI32Constant(op.getLoc(), kUnlockedState, rewriter);
    auto one = createI32Constant(op.getLoc(), kLockedState, rewriter);
    auto cmpxchg = mlir::LLVM::AtomicCmpXchgOp::create(
        rewriter, op.getLoc(), ptr, zero, one,
        mlir::LLVM::AtomicOrdering::acquire,
        mlir::LLVM::AtomicOrdering::monotonic);
    auto acquired = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(), cmpxchg.getResult(),
        llvm::ArrayRef<int64_t>{1});
    rewriter.replaceOp(op, acquired.getResult());
    return mlir::success();
  }
};

struct RawMutexUnlockFastLowering
    : public mlir::OpConversionPattern<SyncRawMutexUnlockFastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRawMutexUnlockFastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto loc = op.getLoc();
    auto ptr = getRawMutexPointer(op, adaptor, converter, rewriter);
    auto zero = createI32Constant(loc, kUnlockedState, rewriter);
    auto contended = createI32Constant(loc, kContendedState, rewriter);
    auto previous = mlir::LLVM::AtomicRMWOp::create(
        rewriter, loc, mlir::LLVM::AtomicBinOp::xchg, ptr, zero,
        mlir::LLVM::AtomicOrdering::release);
    auto wasContended = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::eq, previous.getResult(),
        contended);
    rewriter.replaceOp(op, wasContended.getResult());
    return mlir::success();
  }
};

struct RawRwLockInitLowering
    : public mlir::OpConversionPattern<SyncRawRwLockInitOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRawRwLockInitOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto loc = op.getLoc();
    auto rawRwLockType = converter.convertType(
        llvm::cast<mlir::MemRefType>(op.getRwlock().getType()).getElementType());
    if (!rawRwLockType)
      return rewriter.notifyMatchFailure(op, "failed to convert raw rwlock type");
    auto ptr = getRawRwLockPointer(op, adaptor, converter, rewriter);
    auto statePtr = getRawRwLockStatePointer(loc, ptr, rawRwLockType, rewriter);
    auto writerNotifyPtr =
        getRawRwLockWriterNotifyPointer(loc, ptr, rawRwLockType, rewriter);
    auto zero = createI32Constant(loc, kUnlockedState, rewriter);
    mlir::LLVM::StoreOp::create(rewriter, loc, zero, statePtr);
    mlir::LLVM::StoreOp::create(rewriter, loc, zero, writerNotifyPtr);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct OnceInitLowering : public mlir::OpConversionPattern<SyncOnceInitOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncOnceInitOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto ptr = getOncePointer(op, adaptor, converter, rewriter);
    auto zero = createI32Constant(op.getLoc(), kOnceIncomplete, rewriter);
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), zero, ptr);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct OnceCompletedLowering
    : public mlir::OpConversionPattern<SyncOnceCompletedOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncOnceCompletedOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto ptr = getOncePointer(op, adaptor, converter, rewriter);
    auto state = mlir::LLVM::LoadOp::create(rewriter, op.getLoc(),
                                            rewriter.getI32Type(), ptr);
    auto complete = createI32Constant(op.getLoc(), kOnceComplete, rewriter);
    auto isComplete = mlir::LLVM::ICmpOp::create(
        rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::eq,
        state.getResult(), complete);
    rewriter.replaceOp(op, isComplete.getResult());
    return mlir::success();
  }
};

struct RawRwLockLoadStateLowering
    : public mlir::OpConversionPattern<SyncRawRwLockLoadStateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRawRwLockLoadStateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto loc = op.getLoc();
    auto rawRwLockType = converter.convertType(
        llvm::cast<mlir::MemRefType>(op.getRwlock().getType()).getElementType());
    if (!rawRwLockType)
      return rewriter.notifyMatchFailure(op, "failed to convert raw rwlock type");
    auto ptr = getRawRwLockPointer(op, adaptor, converter, rewriter);
    auto statePtr = getRawRwLockStatePointer(loc, ptr, rawRwLockType, rewriter);
    auto current = mlir::LLVM::LoadOp::create(rewriter, loc, rewriter.getI32Type(),
                                              statePtr);
    rewriter.replaceOp(op, current.getResult());
    return mlir::success();
  }
};

struct RawRwLockReadUnlockFastLowering
    : public mlir::OpConversionPattern<SyncRawRwLockReadUnlockFastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRawRwLockReadUnlockFastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto loc = op.getLoc();
    auto rawRwLockType = converter.convertType(
        llvm::cast<mlir::MemRefType>(op.getRwlock().getType()).getElementType());
    if (!rawRwLockType)
      return rewriter.notifyMatchFailure(op, "failed to convert raw rwlock type");
    auto ptr = getRawRwLockPointer(op, adaptor, converter, rewriter);
    auto statePtr = getRawRwLockStatePointer(loc, ptr, rawRwLockType, rewriter);
    auto readLocked = createI32Constant(loc, kRwLockReadLocked, rewriter);
    auto previous = mlir::LLVM::AtomicRMWOp::create(
        rewriter, loc, mlir::LLVM::AtomicBinOp::sub, statePtr, readLocked,
        mlir::LLVM::AtomicOrdering::release);
    auto state = mlir::LLVM::SubOp::create(rewriter, loc, previous.getResult(),
                                           readLocked);
    rewriter.replaceOp(op, state.getResult());
    return mlir::success();
  }
};

struct RawRwLockCmpxchgStateLowering
    : public mlir::OpConversionPattern<SyncRawRwLockCmpxchgStateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRawRwLockCmpxchgStateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto loc = op.getLoc();
    auto rawRwLockType = converter.convertType(
        llvm::cast<mlir::MemRefType>(op.getRwlock().getType()).getElementType());
    if (!rawRwLockType)
      return rewriter.notifyMatchFailure(op, "failed to convert raw rwlock type");
    auto ptr = getRawRwLockPointer(op, adaptor, converter, rewriter);
    auto statePtr = getRawRwLockStatePointer(loc, ptr, rawRwLockType, rewriter);
    auto cmpxchg = mlir::LLVM::AtomicCmpXchgOp::create(
        rewriter, loc, statePtr, adaptor.getExpected(), adaptor.getDesired(),
        mlir::LLVM::AtomicOrdering::acquire,
        mlir::LLVM::AtomicOrdering::monotonic);
    auto success = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, rewriter.getI1Type(), cmpxchg.getResult(),
        llvm::ArrayRef<int64_t>{1});
    rewriter.replaceOp(op, success.getResult());
    return mlir::success();
  }
};

struct RawRwLockWriteUnlockFastLowering
    : public mlir::OpConversionPattern<SyncRawRwLockWriteUnlockFastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRawRwLockWriteUnlockFastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto loc = op.getLoc();
    auto rawRwLockType = converter.convertType(
        llvm::cast<mlir::MemRefType>(op.getRwlock().getType()).getElementType());
    if (!rawRwLockType)
      return rewriter.notifyMatchFailure(op, "failed to convert raw rwlock type");
    auto ptr = getRawRwLockPointer(op, adaptor, converter, rewriter);
    auto statePtr = getRawRwLockStatePointer(loc, ptr, rawRwLockType, rewriter);
    auto writeLocked = createI32Constant(loc, kRwLockWriteLocked, rewriter);
    auto previous = mlir::LLVM::AtomicRMWOp::create(
        rewriter, loc, mlir::LLVM::AtomicBinOp::sub, statePtr, writeLocked,
        mlir::LLVM::AtomicOrdering::release);
    auto state = mlir::LLVM::SubOp::create(rewriter, loc, previous.getResult(),
                                           writeLocked);
    rewriter.replaceOp(op, state.getResult());
    return mlir::success();
  }
};

struct MutexGetRawMutexLowering
    : public mlir::OpConversionPattern<SyncMutexGetRawMutexOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncMutexGetRawMutexOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto mutexType = llvm::cast<mlir::MemRefType>(op.getMutex().getType());
    auto rawMutexType = llvm::cast<mlir::MemRefType>(op.getRawMutex().getType());
    auto mutexElementType = converter.convertType(mutexType.getElementType());
    if (!mutexElementType)
      return rewriter.notifyMatchFailure(op, "failed to convert mutex type");

    auto mutexPtr =
        getMutexPointer(adaptor.getMutex(), mutexType, converter, op.getLoc(),
                        rewriter);
    auto rawMutexPtr = getMutexFieldPointer(op.getLoc(), mutexPtr,
                                            mutexElementType, /*field=*/0,
                                            rewriter);
    auto descriptor = mlir::MemRefDescriptor::fromStaticShape(
        rewriter, op.getLoc(), converter, rawMutexType, rawMutexPtr);
    rewriter.replaceOp(op, mlir::Value(descriptor));
    return mlir::success();
  }
};

struct MutexGetPayloadLowering
    : public mlir::OpConversionPattern<SyncMutexGetPayloadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncMutexGetPayloadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto mutexType = llvm::cast<mlir::MemRefType>(op.getMutex().getType());
    auto payloadType = llvm::cast<mlir::MemRefType>(op.getPayload().getType());
    auto mutexElementType = converter.convertType(mutexType.getElementType());
    if (!mutexElementType)
      return rewriter.notifyMatchFailure(op, "failed to convert mutex type");

    auto mutexPtr =
        getMutexPointer(adaptor.getMutex(), mutexType, converter, op.getLoc(),
                        rewriter);
    auto payloadPtr = getMutexFieldPointer(op.getLoc(), mutexPtr,
                                           mutexElementType, /*field=*/1,
                                           rewriter);
    auto descriptor = mlir::MemRefDescriptor::fromStaticShape(
        rewriter, op.getLoc(), converter, payloadType, payloadPtr);
    rewriter.replaceOp(op, mlir::Value(descriptor));
    return mlir::success();
  }
};

struct RwLockGetRawRwLockLowering
    : public mlir::OpConversionPattern<SyncRwLockGetRawRwLockOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRwLockGetRawRwLockOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto rwlockType = llvm::cast<mlir::MemRefType>(op.getRwlock().getType());
    auto rawRwLockType =
        llvm::cast<mlir::MemRefType>(op.getRawRwLock().getType());
    auto rwlockElementType = converter.convertType(rwlockType.getElementType());
    if (!rwlockElementType)
      return rewriter.notifyMatchFailure(op, "failed to convert rwlock type");

    auto rwlockPtr =
        getMutexPointer(adaptor.getRwlock(), rwlockType, converter, op.getLoc(),
                        rewriter);
    auto rawRwLockPtr = getMutexFieldPointer(op.getLoc(), rwlockPtr,
                                             rwlockElementType, /*field=*/0,
                                             rewriter);
    auto descriptor = mlir::MemRefDescriptor::fromStaticShape(
        rewriter, op.getLoc(), converter, rawRwLockType, rawRwLockPtr);
    rewriter.replaceOp(op, mlir::Value(descriptor));
    return mlir::success();
  }
};

struct RwLockGetPayloadLowering
    : public mlir::OpConversionPattern<SyncRwLockGetPayloadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRwLockGetPayloadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto rwlockType = llvm::cast<mlir::MemRefType>(op.getRwlock().getType());
    auto payloadType = llvm::cast<mlir::MemRefType>(op.getPayload().getType());
    auto rwlockElementType = converter.convertType(rwlockType.getElementType());
    if (!rwlockElementType)
      return rewriter.notifyMatchFailure(op, "failed to convert rwlock type");

    auto rwlockPtr =
        getMutexPointer(adaptor.getRwlock(), rwlockType, converter, op.getLoc(),
                        rewriter);
    auto payloadPtr = getMutexFieldPointer(op.getLoc(), rwlockPtr,
                                           rwlockElementType, /*field=*/1,
                                           rewriter);
    auto descriptor = mlir::MemRefDescriptor::fromStaticShape(
        rewriter, op.getLoc(), converter, payloadType, payloadPtr);
    rewriter.replaceOp(op, mlir::Value(descriptor));
    return mlir::success();
  }
};

struct CombiningLockInitLowering
    : public mlir::OpConversionPattern<SyncCombiningLockInitOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncCombiningLockInitOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto lockType = llvm::cast<mlir::MemRefType>(op.getLock().getType());
    auto lockElementType = converter.convertType(lockType.getElementType());
    if (!lockElementType)
      return rewriter.notifyMatchFailure(op, "failed to convert combining lock");

    auto lockPtr = getCombiningLockPointer(op, adaptor.getLock(), converter, rewriter);
    auto nullPtr = mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(),
                                              getOpaquePtrType(op.getContext()));
    auto zeroI8 = createI8Constant(op.getLoc(), 0, rewriter);

    auto tailPtr = getCombiningLockFieldPointer(op.getLoc(), lockPtr, lockElementType,
                                                {0, 0, 0}, rewriter);
    auto statusPtr = getCombiningLockFieldPointer(op.getLoc(), lockPtr, lockElementType,
                                                  {0, 0, 1}, rewriter);
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), nullPtr.getResult(), tailPtr);
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), zeroI8, statusPtr);

    if (mlir::Value initialValue = adaptor.getInitialValue()) {
      auto payloadPtr = getCombiningLockFieldPointer(op.getLoc(), lockPtr,
                                                     lockElementType, {0, 1},
                                                     rewriter);
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), initialValue, payloadPtr);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct CombiningLockHasTailLowering
    : public mlir::OpConversionPattern<SyncCombiningLockHasTailOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncCombiningLockHasTailOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto lockType = llvm::cast<mlir::MemRefType>(op.getLock().getType());
    auto lockElementType = converter.convertType(lockType.getElementType());
    if (!lockElementType)
      return rewriter.notifyMatchFailure(op, "failed to convert combining lock");

    auto loc = op.getLoc();
    auto lockPtr = getCombiningLockPointer(op, adaptor.getLock(), converter, rewriter);
    auto tailPtr = getCombiningLockFieldPointer(loc, lockPtr, lockElementType,
                                                {0, 0, 0}, rewriter);
    unsigned ptrAlignment = converter.getPointerBitwidth(0) / 8;
    auto tail = mlir::LLVM::LoadOp::create(rewriter, loc, getOpaquePtrType(op.getContext()),
                                           tailPtr, ptrAlignment, false, false, false, false,
                                           mlir::LLVM::AtomicOrdering::monotonic);
    auto nullPtr = mlir::LLVM::ZeroOp::create(rewriter, loc,
                                              getOpaquePtrType(op.getContext()));
    auto hasTail = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::ne, tail.getResult(),
        nullPtr.getResult());
    rewriter.replaceOp(op, hasTail.getResult());
    return mlir::success();
  }
};

struct CombiningLockTryAcquireLowering
    : public mlir::OpConversionPattern<SyncCombiningLockTryAcquireOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncCombiningLockTryAcquireOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto lockType = llvm::cast<mlir::MemRefType>(op.getLock().getType());
    auto lockElementType = converter.convertType(lockType.getElementType());
    if (!lockElementType)
      return rewriter.notifyMatchFailure(op, "failed to convert combining lock");

    auto loc = op.getLoc();
    auto lockPtr = getCombiningLockPointer(op, adaptor.getLock(), converter, rewriter);
    auto statusPtr = getCombiningLockFieldPointer(loc, lockPtr, lockElementType,
                                                  {0, 0, 1}, rewriter);
    auto zero = createI8Constant(loc, 0, rewriter);
    auto one = createI8Constant(loc, 1, rewriter);
    auto previous = mlir::LLVM::AtomicRMWOp::create(
        rewriter, loc, mlir::LLVM::AtomicBinOp::xchg, statusPtr, one,
        mlir::LLVM::AtomicOrdering::acquire,
        /*syncscope=*/"", /*alignment=*/1);
    auto acquired = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::eq, previous.getResult(),
        zero);
    rewriter.replaceOp(op, acquired.getResult());
    return mlir::success();
  }
};

struct CombiningLockReleaseLowering
    : public mlir::OpConversionPattern<SyncCombiningLockReleaseOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncCombiningLockReleaseOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto lockType = llvm::cast<mlir::MemRefType>(op.getLock().getType());
    auto lockElementType = converter.convertType(lockType.getElementType());
    if (!lockElementType)
      return rewriter.notifyMatchFailure(op, "failed to convert combining lock");

    auto lockPtr = getCombiningLockPointer(op, adaptor.getLock(), converter, rewriter);
    auto statusPtr = getCombiningLockFieldPointer(op.getLoc(), lockPtr, lockElementType,
                                                  {0, 0, 1}, rewriter);
    auto zero = createI8Constant(op.getLoc(), 0, rewriter);
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), zero, statusPtr,
                                /*alignment=*/1, /*isVolatile=*/false,
                                /*isNonTemporal=*/false,
                                /*isInvariantGroup=*/false,
                                mlir::LLVM::AtomicOrdering::release);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct CombiningLockGetPayloadLowering
    : public mlir::OpConversionPattern<SyncCombiningLockGetPayloadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncCombiningLockGetPayloadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto sourceType = llvm::cast<mlir::MemRefType>(op.getSource().getType());
    auto sourceElementType = sourceType.getElementType();
    auto loweredSourceType = converter.convertType(sourceElementType);
    if (!loweredSourceType)
      return rewriter.notifyMatchFailure(op, "failed to convert source type");

    if (llvm::isa<CombiningLockType>(sourceElementType)) {
      auto payloadType = llvm::cast<mlir::MemRefType>(op.getResult(0).getType());
      auto sourcePtr =
          mlir::LLVM::getStridedElementPtr(rewriter, op.getLoc(), converter,
                                           sourceType, adaptor.getSource(), {});
      auto payloadPtr = getCombiningLockFieldPointer(op.getLoc(), sourcePtr,
                                                     loweredSourceType, {0, 1},
                                                     rewriter);
      rewriter.replaceOp(op, buildStaticZeroRankMemRef(op.getLoc(), converter,
                                                       payloadType, payloadPtr,
                                                       rewriter));
      return mlir::success();
    }

    auto nodeType = llvm::cast<CombiningLockNodeType>(sourceElementType);
    auto nodePtr = getNodePointer(op, adaptor.getSource(), sourceType, converter,
                                  rewriter);
    auto payloadFieldPtr = getCombiningLockFieldPointer(op.getLoc(), nodePtr,
                                                        loweredSourceType, {0, 3},
                                                        rewriter);
    auto payloadRawPtr = mlir::LLVM::LoadOp::create(
        rewriter, op.getLoc(), getOpaquePtrType(op.getContext()), payloadFieldPtr);

    llvm::SmallVector<mlir::Value> replacements;
    replacements.reserve(op.getNumResults());

    auto payloadType = llvm::cast<mlir::MemRefType>(op.getResult(0).getType());
    replacements.push_back(buildStaticZeroRankMemRef(op.getLoc(), converter,
                                                     payloadType,
                                                     payloadRawPtr.getResult(),
                                                     rewriter));

    for (auto [index, captureType] : llvm::enumerate(nodeType.getCaptureTypes())) {
      auto loweredCaptureType = converter.convertType(captureType);
      if (!loweredCaptureType)
        return rewriter.notifyMatchFailure(op, "failed to convert capture type");
      auto captureFieldPtr = getCombiningLockFieldPointer(
          op.getLoc(), nodePtr, loweredSourceType,
          {0, static_cast<int32_t>(index + 4)}, rewriter);
      auto capture = mlir::LLVM::LoadOp::create(rewriter, op.getLoc(),
                                                loweredCaptureType,
                                                captureFieldPtr);
      replacements.push_back(capture.getResult());
    }

    rewriter.replaceOp(op, replacements);
    return mlir::success();
  }
};

struct CombiningLockCaptureLowering
    : public mlir::OpConversionPattern<SyncCombiningLockCaptureOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncCombiningLockCaptureOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto nodeMemRefType = llvm::cast<mlir::MemRefType>(op.getNode().getType());
    auto nodeElementType = converter.convertType(nodeMemRefType.getElementType());
    if (!nodeElementType)
      return rewriter.notifyMatchFailure(op, "failed to convert node type");

    auto arraySize = createI64Constant(op.getLoc(), 1, rewriter);
    auto alloca = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), getOpaquePtrType(op.getContext()), nodeElementType,
        arraySize);
    mlir::LLVM::LifetimeStartOp::create(rewriter, op.getLoc(), alloca.getResult());

    auto futexPtr = getCombiningLockFieldPointer(op.getLoc(), alloca.getResult(),
                                                 nodeElementType, {0, 0},
                                                 rewriter);
    auto nextPtr = getCombiningLockFieldPointer(op.getLoc(), alloca.getResult(),
                                                nodeElementType, {0, 1},
                                                rewriter);
    auto closurePtr = getCombiningLockFieldPointer(op.getLoc(), alloca.getResult(),
                                                   nodeElementType, {0, 2},
                                                   rewriter);
    auto payloadPtrField = getCombiningLockFieldPointer(
        op.getLoc(), alloca.getResult(), nodeElementType, {0, 3}, rewriter);

    auto waiting = createI32Constant(op.getLoc(), kCombiningNodeWaiting, rewriter);
    auto nullPtr = mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(),
                                              getOpaquePtrType(op.getContext()));
    auto payloadType = llvm::cast<mlir::MemRefType>(op.getPayload().getType());
    auto payloadPtr = mlir::LLVM::getStridedElementPtr(
        rewriter, op.getLoc(), converter, payloadType, adaptor.getPayload(), {});

    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), waiting, futexPtr);
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), nullPtr.getResult(), nextPtr);
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getCallee(),
                                closurePtr,
                                /*alignment=*/0, /*isVolatile=*/false,
                                /*isNonTemporal=*/false,
                                /*isInvariantGroup=*/true);
    mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), payloadPtr, payloadPtrField);

    auto nodeType = llvm::cast<CombiningLockNodeType>(nodeMemRefType.getElementType());
    for (auto [index, captureType, captureValue] :
         llvm::zip_equal(llvm::seq<size_t>(0, nodeType.getCaptureTypes().size()),
                         nodeType.getCaptureTypes(), adaptor.getCaptures())) {
      auto loweredCaptureType = converter.convertType(captureType);
      if (!loweredCaptureType)
        return rewriter.notifyMatchFailure(op, "failed to convert capture type");
      auto captureFieldPtr = getCombiningLockFieldPointer(
          op.getLoc(), alloca.getResult(), nodeElementType,
          {0, static_cast<int32_t>(index + 4)}, rewriter);
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), captureValue, captureFieldPtr,
                                  /*alignment=*/0, /*isVolatile=*/false,
                                  /*isNonTemporal=*/false,
                                  /*isInvariantGroup=*/true);
    }

    auto nodeDescriptor = buildStaticZeroRankMemRef(op.getLoc(), converter,
                                                    nodeMemRefType,
                                                    alloca.getResult(), rewriter);
    rewriter.replaceOp(op, mlir::ValueRange{nodeDescriptor, alloca.getResult()});
    return mlir::success();
  }
};

struct CombiningLockCaptureEndLowering
    : public mlir::OpConversionPattern<SyncCombiningLockCaptureEndOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncCombiningLockCaptureEndOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::LLVM::LifetimeEndOp::create(rewriter, op.getLoc(), adaptor.getRawNode());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct CombiningLockRecoverLowering
    : public mlir::OpConversionPattern<SyncCombiningLockRecoverOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncCombiningLockRecoverOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter =
        *static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    auto captureTypes = llvm::to_vector(llvm::drop_begin(op.getResultTypes()));
    auto nodeElementType =
        getCombiningNodeLLVMTypeForCaptures(converter, op.getContext(),
                                            captureTypes);
    if (!nodeElementType)
      return rewriter.notifyMatchFailure(op, "failed to convert recovery node type");

    auto payloadFieldPtr = getCombiningLockFieldPointer(op.getLoc(), adaptor.getNode(),
                                                        nodeElementType, {0, 3},
                                                        rewriter);
    auto payloadRawPtr = mlir::LLVM::LoadOp::create(
        rewriter, op.getLoc(), getOpaquePtrType(op.getContext()), payloadFieldPtr);

    llvm::SmallVector<mlir::Value> replacements;
    replacements.reserve(op.getNumResults());

    auto payloadType = llvm::cast<mlir::MemRefType>(op.getResult(0).getType());
    replacements.push_back(buildStaticZeroRankMemRef(op.getLoc(), converter,
                                                     payloadType,
                                                     payloadRawPtr.getResult(),
                                                     rewriter));

    for (auto [index, captureType] :
         llvm::enumerate(llvm::drop_begin(op.getResultTypes()))) {
      auto loweredCaptureType = converter.convertType(captureType);
      if (!loweredCaptureType)
        return rewriter.notifyMatchFailure(op, "failed to convert capture type");
      auto captureFieldPtr = getCombiningLockFieldPointer(
          op.getLoc(), adaptor.getNode(), nodeElementType,
          {0, static_cast<int32_t>(index + 4)}, rewriter);
      auto capture = mlir::LLVM::LoadOp::create(rewriter, op.getLoc(),
                                                loweredCaptureType,
                                                captureFieldPtr,
                                                /*alignment=*/0,
                                                /*isVolatile=*/false,
                                                /*isNonTemporal=*/false,
                                                /*isInvariant=*/false,
                                                /*isInvariantGroup=*/true);
      replacements.push_back(capture.getResult());
    }

    rewriter.replaceOp(op, replacements);
    return mlir::success();
  }
};

} // namespace

void configureConvertSyncToLLVMConversionLegality(
    mlir::ConversionTarget &target) {
  target.addIllegalDialect<SyncDialect>();
}

void populateConvertSyncToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<RawMutexInitLowering, RawMutexTryLockLowering,
               RawMutexUnlockFastLowering, RawRwLockInitLowering,
               OnceInitLowering, OnceCompletedLowering,
               RawRwLockLoadStateLowering, RawRwLockCmpxchgStateLowering,
               RawRwLockReadUnlockFastLowering,
               RawRwLockWriteUnlockFastLowering, MutexGetRawMutexLowering,
               MutexGetPayloadLowering, RwLockGetRawRwLockLowering,
               RwLockGetPayloadLowering, CombiningLockInitLowering,
               CombiningLockHasTailLowering, CombiningLockTryAcquireLowering,
               CombiningLockReleaseLowering,
               CombiningLockGetPayloadLowering, CombiningLockCaptureLowering,
               CombiningLockCaptureEndLowering,
               CombiningLockRecoverLowering>(
      converter, patterns.getContext());
}

namespace {

struct SyncConvertToLLVMPatternInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const override {
    populateSyncToLLVMTypeConversions(typeConverter);
    configureConvertSyncToLLVMConversionLegality(target);
    populateConvertSyncToLLVMConversionPatterns(typeConverter, patterns);
  }
};

struct ConvertSyncToLLVMPass
    : public impl::ConvertSyncToLLVMPassBase<ConvertSyncToLLVMPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    LLVMTypeConverter converter(moduleOp);
    mlir::RewritePatternSet patterns(&getContext());
    populateConvertSyncToLLVMConversionPatterns(converter, patterns);

    mlir::ConversionTarget target(getContext());
    configureConvertSyncToLLVMConversionLegality(target);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    if (mlir::failed(
            mlir::applyFullConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void registerConvertSyncToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context, SyncDialect *dialect) {
    dialect->addInterfaces<SyncConvertToLLVMPatternInterface>();
  });
}

} // namespace mlir::sync
