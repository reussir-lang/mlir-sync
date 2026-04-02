#include "Sync/Conversion/ConvertSyncToLLVM.h"

#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
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
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(value));
}

template <typename OpTy>
mlir::Value getRawMutexPointer(OpTy op, typename OpTy::Adaptor adaptor,
                               const LLVMTypeConverter &converter,
                               mlir::ConversionPatternRewriter &rewriter) {
  auto memrefType = llvm::cast<mlir::MemRefType>(op.getMutex().getType());
  return mlir::LLVM::getStridedElementPtr(rewriter, op.getLoc(), converter,
                                          memrefType, adaptor.getMutex(), {});
}

struct RawMutexInitLowering
    : public mlir::OpConversionPattern<SyncRawMutexInitOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SyncRawMutexInitOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *static_cast<const LLVMTypeConverter *>(getTypeConverter());
    auto ptr = getRawMutexPointer(op, adaptor, converter, rewriter);
    auto zero = createI32Constant(op.getLoc(), kUnlockedState, rewriter);
    rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), zero, ptr);
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
    auto &converter = *static_cast<const LLVMTypeConverter *>(getTypeConverter());
    auto ptr = getRawMutexPointer(op, adaptor, converter, rewriter);
    auto zero = createI32Constant(op.getLoc(), kUnlockedState, rewriter);
    auto one = createI32Constant(op.getLoc(), kLockedState, rewriter);
    auto cmpxchg = rewriter.create<mlir::LLVM::AtomicCmpXchgOp>(
        op.getLoc(), ptr, zero, one, mlir::LLVM::AtomicOrdering::acquire,
        mlir::LLVM::AtomicOrdering::monotonic);
    auto acquired = rewriter.create<mlir::LLVM::ExtractValueOp>(
        op.getLoc(), rewriter.getI1Type(), cmpxchg.getResult(),
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
    auto &converter = *static_cast<const LLVMTypeConverter *>(getTypeConverter());
    auto loc = op.getLoc();
    auto ptr = getRawMutexPointer(op, adaptor, converter, rewriter);
    auto zero = createI32Constant(loc, kUnlockedState, rewriter);
    auto contended = createI32Constant(loc, kContendedState, rewriter);
    auto previous = rewriter.create<mlir::LLVM::AtomicRMWOp>(
        loc, mlir::LLVM::AtomicBinOp::xchg, ptr, zero,
        mlir::LLVM::AtomicOrdering::release);
    auto wasContended = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, previous.getResult(), contended);
    rewriter.replaceOp(op, wasContended.getResult());
    return mlir::success();
  }
};

} // namespace

void populateConvertSyncToLLVMConversionPatterns(
    LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<RawMutexInitLowering, RawMutexTryLockLowering,
               RawMutexUnlockFastLowering>(
      converter, patterns.getContext());
}

namespace {

struct ConvertSyncToLLVMPass
    : public impl::ConvertSyncToLLVMPassBase<ConvertSyncToLLVMPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    LLVMTypeConverter converter(moduleOp);
    mlir::RewritePatternSet patterns(&getContext());
    mlir::SymbolTableCollection symbolTables;
    populateConvertSyncToLLVMConversionPatterns(converter, patterns);
    mlir::ptr::populatePtrToLLVMConversionPatterns(converter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(converter, patterns,
                                               &symbolTables);
    mlir::populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns,
                                                         &symbolTables);

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::BuiltinDialect, mlir::cf::ControlFlowDialect,
                           mlir::LLVM::LLVMDialect>();
    target.addIllegalDialect<SyncDialect, mlir::func::FuncDialect,
                             mlir::memref::MemRefDialect, mlir::ptr::PtrDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    if (mlir::failed(
            mlir::applyFullConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::sync
