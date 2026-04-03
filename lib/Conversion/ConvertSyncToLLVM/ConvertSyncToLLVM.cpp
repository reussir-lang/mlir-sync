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

mlir::Value createI32Constant(mlir::Location loc, uint64_t value,
                              mlir::ConversionPatternRewriter &rewriter) {
  return mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                        rewriter.getI32IntegerAttr(value));
}

template <typename OpTy>
mlir::Value getRawMutexPointer(OpTy op, typename OpTy::Adaptor adaptor,
                               const mlir::LLVMTypeConverter &converter,
                               mlir::ConversionPatternRewriter &rewriter) {
  auto memrefType = llvm::cast<mlir::MemRefType>(op.getMutex().getType());
  return mlir::LLVM::getStridedElementPtr(rewriter, op.getLoc(), converter,
                                          memrefType, adaptor.getMutex(), {});
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

} // namespace

void configureConvertSyncToLLVMConversionLegality(
    mlir::ConversionTarget &target) {
  target.addIllegalDialect<SyncDialect>();
}

void populateConvertSyncToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<RawMutexInitLowering, RawMutexTryLockLowering,
               RawMutexUnlockFastLowering, MutexGetRawMutexLowering,
               MutexGetPayloadLowering>(
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
