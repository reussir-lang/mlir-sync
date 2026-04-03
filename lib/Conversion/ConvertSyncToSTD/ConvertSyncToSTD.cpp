#include "Sync/Conversion/ConvertSyncToSTD.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "Sync/IR/SyncOps.h"

namespace mlir::sync {

#define GEN_PASS_DEF_CONVERTSYNCTOSTDPASS
#include "Sync/Conversion/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kLockSlowPath = "mlir_sync_mutex_lock_slow_path";
constexpr llvm::StringLiteral kUnlockSlowPath =
    "mlir_sync_mutex_unlock_slow_path";

mlir::func::FuncOp getOrCreateRuntimeFunc(mlir::Location loc,
                                          llvm::StringRef name,
                                          mlir::Type argumentType,
                                          mlir::TypeRange resultTypes,
                                          mlir::ModuleOp moduleOp,
                                          mlir::PatternRewriter &rewriter) {
  if (auto func = moduleOp.lookupSymbol<mlir::func::FuncOp>(name))
    return func;

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto fnType = rewriter.getFunctionType(argumentType, resultTypes);
  auto func = mlir::func::FuncOp::create(rewriter, loc, name, fnType);
  func.setPrivate();
  return func;
}

mlir::ptr::PtrType getRuntimePtrType(mlir::Value mutex) {
  auto memrefType = llvm::cast<mlir::MemRefType>(mutex.getType());
  auto memorySpace =
      llvm::dyn_cast_or_null<mlir::ptr::MemorySpaceAttrInterface>(
          memrefType.getMemorySpace());
  if (!memorySpace)
    memorySpace = mlir::ptr::GenericSpaceAttr::get(mutex.getContext());
  return mlir::ptr::PtrType::get(mutex.getContext(), memorySpace);
}

mlir::Value materializeRuntimePtr(mlir::Location loc, mlir::Value mutex,
                                  mlir::ptr::PtrType ptrType,
                                  mlir::PatternRewriter &rewriter) {
  auto memrefType = llvm::cast<mlir::MemRefType>(mutex.getType());
  if (memrefType.getMemorySpace() != ptrType.getMemorySpace()) {
    auto castedType = mlir::MemRefType::get(
        memrefType.getShape(), memrefType.getElementType(), memrefType.getLayout(),
        ptrType.getMemorySpace());
    mutex =
        mlir::memref::MemorySpaceCastOp::create(rewriter, loc, castedType, mutex);
  }
  return mlir::ptr::ToPtrOp::create(rewriter, loc, ptrType, mutex);
}

struct RawMutexLockLowering : public mlir::OpRewritePattern<SyncRawMutexLockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRawMutexLockOp op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto ptrType = getRuntimePtrType(op.getMutex());
    auto slowPathFunc =
        getOrCreateRuntimeFunc(loc, kLockSlowPath, ptrType, {}, moduleOp,
                               rewriter);
    auto acquired = SyncRawMutexTryLockOp::create(rewriter, loc, rewriter.getI1Type(),
                                                  op.getMutex())
                        .getResult();
    auto ifOp = mlir::scf::IfOp::create(rewriter, loc, acquired,
                                        /*withElseRegion=*/true);

    rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    auto ptr = materializeRuntimePtr(loc, op.getMutex(), ptrType, rewriter);
    mlir::func::CallOp::create(rewriter, loc, slowPathFunc, mlir::ValueRange{ptr});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RawMutexUnlockLowering
    : public mlir::OpRewritePattern<SyncRawMutexUnlockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRawMutexUnlockOp op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto ptrType = getRuntimePtrType(op.getMutex());
    auto unlockSlowFunc =
        getOrCreateRuntimeFunc(loc, kUnlockSlowPath, ptrType, {}, moduleOp,
                               rewriter);
    auto needsWake = SyncRawMutexUnlockFastOp::create(rewriter, loc,
                                                      rewriter.getI1Type(),
                                                      op.getMutex())
                         .getResult();
    auto ifOp = mlir::scf::IfOp::create(rewriter, loc, needsWake,
                                        /*withElseRegion=*/false);

    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    auto slowPathPtr = materializeRuntimePtr(loc, op.getMutex(), ptrType, rewriter);
    mlir::func::CallOp::create(rewriter, loc, unlockSlowFunc,
                               mlir::ValueRange{slowPathPtr});

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

void populateConvertSyncToSTDPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<RawMutexLockLowering, RawMutexUnlockLowering>(
      patterns.getContext());
}

namespace {

struct ConvertSyncToSTDPass
    : public impl::ConvertSyncToSTDPassBase<ConvertSyncToSTDPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    populateConvertSyncToSTDPatterns(patterns);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::sync
