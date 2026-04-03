#include "Sync/Conversion/ConvertSyncToSTD.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/RegionUtils.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <llvm/ADT/SetVector.h>

#include "Sync/IR/SyncOps.h"

namespace mlir::sync {

#define GEN_PASS_DEF_CONVERTSYNCTOSTDPASS
#include "Sync/Conversion/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kLockSlowPath = "mlir_sync_mutex_lock_slow_path";
constexpr llvm::StringLiteral kUnlockSlowPath =
    "mlir_sync_mutex_unlock_slow_path";
constexpr llvm::StringLiteral kCombiningAttachSlowPath =
    "mlir_sync_combining_lock_attach_slow_path";
constexpr llvm::StringLiteral kCombiningWrapperPrefix =
    "__sync_combining_lock_slow_";

mlir::func::FuncOp getOrCreateRuntimeFunc(mlir::Location loc,
                                          llvm::StringRef name,
                                          mlir::TypeRange argumentTypes,
                                          mlir::TypeRange resultTypes,
                                          mlir::ModuleOp moduleOp,
                                          mlir::PatternRewriter &rewriter) {
  if (auto func = moduleOp.lookupSymbol<mlir::func::FuncOp>(name))
    return func;

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto fnType = rewriter.getFunctionType(argumentTypes, resultTypes);
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

mlir::MemRefType getCombiningLockPayloadProjectionType(mlir::MemRefType lockType) {
  auto lockElementType =
      llvm::cast<CombiningLockType>(lockType.getElementType());
  return mlir::MemRefType::get({}, lockElementType.getValueType(),
                               mlir::MemRefLayoutAttrInterface{},
                               lockType.getMemorySpace());
}

mlir::MemRefType getGenericNodeMemRefType(mlir::MLIRContext *context,
                                          CombiningLockNodeType nodeType) {
  return mlir::MemRefType::get(
      {}, nodeType, mlir::MemRefLayoutAttrInterface{},
      mlir::ptr::GenericSpaceAttr::get(context));
}

std::string getUniqueFunctionName(mlir::ModuleOp moduleOp, llvm::StringRef prefix) {
  unsigned ordinal = 0;
  while (true) {
    std::string name = (prefix + llvm::Twine(ordinal++)).str();
    if (!moduleOp.lookupSymbol(name))
      return name;
  }
}

mlir::func::FuncOp createCombiningLockWrapper(
    SyncCombiningLockCriticalSectionOp op, mlir::ModuleOp moduleOp,
    mlir::MemRefType payloadType, mlir::ArrayRef<mlir::Value> captures,
    mlir::PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto ptrType = mlir::ptr::PtrType::get(
      op.getContext(), mlir::ptr::GenericSpaceAttr::get(op.getContext()));
  auto wrapperType = rewriter.getFunctionType({ptrType}, {});
  std::string name = getUniqueFunctionName(moduleOp, kCombiningWrapperPrefix);
  op.getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto wrapper = mlir::func::FuncOp::create(rewriter, loc, name, wrapperType);
  wrapper.setPrivate();
  wrapper.setNoInline(true);
  wrapper->setAttr("llvm.linkage", mlir::LLVM::LinkageAttr::get(
                                       op.getContext(),
                                       mlir::LLVM::linkage::Linkage::Internal));
  wrapper->setAttr("passthrough",
                   rewriter.getArrayAttr({rewriter.getStringAttr("cold")}));

  auto *entryBlock = wrapper.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  llvm::SmallVector<mlir::Type> captureTypes;
  captureTypes.reserve(captures.size());
  for (mlir::Value capture : captures)
    captureTypes.push_back(capture.getType());

  llvm::SmallVector<mlir::Type> recoveredTypes;
  recoveredTypes.push_back(payloadType);
  llvm::append_range(recoveredTypes, captureTypes);
  auto recovered = SyncCombiningLockRecoverOp::create(
      rewriter, loc, recoveredTypes, entryBlock->getArgument(0));

  mlir::IRMapping mapping;
  auto &bodyBlock = op.getBody().front();
  mapping.map(bodyBlock.getArgument(0), recovered.getResult(0));
  for (auto [capture, recoveredValue] :
       llvm::zip_equal(captures, recovered.getResults().drop_front()))
    mapping.map(capture, recoveredValue);

  for (mlir::Operation &nestedOp : bodyBlock.without_terminator())
    rewriter.clone(nestedOp, mapping);

  mlir::func::ReturnOp::create(rewriter, loc);
  return wrapper;
}

void cloneCombiningLockBody(SyncCombiningLockCriticalSectionOp op,
                            mlir::Value payload,
                            mlir::PatternRewriter &rewriter) {
  mlir::IRMapping mapping;
  auto &bodyBlock = op.getBody().front();
  mapping.map(bodyBlock.getArgument(0), payload);
  for (mlir::Operation &nestedOp : bodyBlock.without_terminator())
    rewriter.clone(nestedOp, mapping);
}

void emitCombiningLockSlowPath(
    SyncCombiningLockCriticalSectionOp op, mlir::Value payload,
    mlir::ArrayRef<mlir::Value> captures, mlir::func::FuncOp wrapper,
    mlir::ModuleOp moduleOp, mlir::PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto nodeType = CombiningLockNodeType::get(
      op.getContext(), llvm::to_vector(llvm::map_range(
                           captures, [](mlir::Value value) {
                             return value.getType();
                           })));
  auto nodeMemRefType = getGenericNodeMemRefType(op.getContext(), nodeType);
  auto wrapperValue = mlir::func::ConstantOp::create(
                          rewriter, loc, wrapper.getFunctionType(),
                          wrapper.getSymName())
                          .getResult();
  llvm::SmallVector<mlir::Value> captureOperands{wrapperValue, payload};
  llvm::append_range(captureOperands, captures);
  auto rawNodeType = mlir::ptr::PtrType::get(
      op.getContext(), mlir::ptr::GenericSpaceAttr::get(op.getContext()));
  auto capture = SyncCombiningLockCaptureOp::create(
      rewriter, loc, mlir::TypeRange{nodeMemRefType, rawNodeType},
      captureOperands);
  auto node = capture.getNode();
  auto rawNode = capture.getRawNode();

  auto nodePtrType = getRuntimePtrType(node);
  auto lockPtrType = getRuntimePtrType(op.getLock());
  auto attachFunc = getOrCreateRuntimeFunc(
      loc, kCombiningAttachSlowPath,
      {nodePtrType, lockPtrType, rewriter.getI64Type()}, {}, moduleOp,
      rewriter);
  auto nodePtr = rawNode;
  if (rawNode.getType() != nodePtrType)
    nodePtr = mlir::ptr::ToPtrOp::create(rewriter, loc, nodePtrType, rawNode);
  auto lockPtr = materializeRuntimePtr(loc, op.getLock(), lockPtrType, rewriter);
  int64_t combineLimit = -1;
  if (auto combineLimitAttr = op.getCombineLimitAttr())
    combineLimit = combineLimitAttr.getInt();
  auto combineLimitValue =
      mlir::arith::ConstantIntOp::create(rewriter, loc, combineLimit, 64);
  mlir::func::CallOp::create(rewriter, loc, attachFunc,
                             mlir::ValueRange{nodePtr, lockPtr,
                                              combineLimitValue.getResult()});
  SyncCombiningLockCaptureEndOp::create(rewriter, loc, rawNode);
}

struct RawMutexLockLowering : public mlir::OpRewritePattern<SyncRawMutexLockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRawMutexLockOp op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto ptrType = getRuntimePtrType(op.getMutex());
    auto slowPathFunc =
        getOrCreateRuntimeFunc(loc, kLockSlowPath, {ptrType}, {}, moduleOp,
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

struct MutexInitLowering : public mlir::OpRewritePattern<SyncMutexInitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncMutexInitOp op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mutexType = llvm::cast<mlir::MemRefType>(op.getMutex().getType());
    auto rawMutexType = getRawMutexProjectionType(mutexType);
    auto rawMutex =
        SyncMutexGetRawMutexOp::create(rewriter, loc, rawMutexType, op.getMutex())
            .getResult();
    SyncRawMutexInitOp::create(rewriter, loc, rawMutex);
    if (mlir::Value initialValue = op.getInitialValue()) {
      auto payloadType = getPayloadProjectionType(mutexType);
      auto payload =
          SyncMutexGetPayloadOp::create(rewriter, loc, payloadType, op.getMutex())
              .getResult();
      mlir::memref::StoreOp::create(rewriter, loc, initialValue, payload,
                                    mlir::ValueRange{});
    }
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
        getOrCreateRuntimeFunc(loc, kUnlockSlowPath, {ptrType}, {}, moduleOp,
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

struct MutexCriticalSectionLowering
    : public mlir::OpRewritePattern<SyncMutexCriticalSectionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncMutexCriticalSectionOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mutexType = llvm::cast<mlir::MemRefType>(op.getMutex().getType());
    auto rawMutexType = getRawMutexProjectionType(mutexType);
    auto payloadType = getPayloadProjectionType(mutexType);

    rewriter.setInsertionPoint(op);
    auto rawMutex =
        SyncMutexGetRawMutexOp::create(rewriter, loc, rawMutexType, op.getMutex())
            .getResult();
    SyncRawMutexLockOp::create(rewriter, loc, rawMutex);
    auto payload =
        SyncMutexGetPayloadOp::create(rewriter, loc, payloadType, op.getMutex())
            .getResult();

    auto &bodyBlock = op.getBody().front();
    auto yieldOp = llvm::cast<SyncYieldOp>(bodyBlock.getTerminator());
    llvm::SmallVector<mlir::Value> yieldedValues;
    yieldedValues.reserve(yieldOp->getNumOperands());
    for (mlir::Value value : yieldOp->getOperands())
      yieldedValues.push_back(value == bodyBlock.getArgument(0) ? payload
                                                                : value);
    rewriter.eraseOp(yieldOp);
    rewriter.inlineBlockBefore(&bodyBlock, op, mlir::ValueRange{payload});
    rewriter.setInsertionPoint(op);
    SyncRawMutexUnlockOp::create(rewriter, loc, rawMutex);

    if (op.getNumResults() == 0)
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, yieldedValues);
    return mlir::success();
  }
};

struct CombiningLockCriticalSectionLowering
    : public mlir::OpRewritePattern<SyncCombiningLockCriticalSectionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncCombiningLockCriticalSectionOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto lockType = llvm::cast<mlir::MemRefType>(op.getLock().getType());
    auto payloadType = getCombiningLockPayloadProjectionType(lockType);

    llvm::SetVector<mlir::Value> capturedValueSet;
    op.getBody().walk([&](mlir::Operation *nestedOp) {
      for (mlir::Value operand : nestedOp->getOperands()) {
        bool definedInBody = false;
        for (mlir::Region *region = operand.getParentRegion(); region;
             region = region->getParentRegion()) {
          if (region == &op.getBody()) {
            definedInBody = true;
            break;
          }
        }
        if (!definedInBody)
          capturedValueSet.insert(operand);
      }
    });
    llvm::SmallVector<mlir::Value> captures(capturedValueSet.begin(),
                                            capturedValueSet.end());

    auto wrapper =
        createCombiningLockWrapper(op, moduleOp, payloadType, captures, rewriter);

    rewriter.setInsertionPoint(op);
    auto payload = SyncCombiningLockGetPayloadOp::create(rewriter, loc, payloadType,
                                                         op.getLock())
                       .getResult(0);
    auto hasTail = SyncCombiningLockHasTailOp::create(
                       rewriter, loc, rewriter.getI1Type(), op.getLock())
                       .getResult();
    auto outerIf =
        mlir::scf::IfOp::create(rewriter, loc, hasTail, /*withElseRegion=*/true);

    rewriter.setInsertionPoint(outerIf.thenBlock()->getTerminator());
    emitCombiningLockSlowPath(op, payload, captures, wrapper, moduleOp, rewriter);

    rewriter.setInsertionPoint(outerIf.elseBlock()->getTerminator());
    auto acquired = SyncCombiningLockTryAcquireOp::create(
                        rewriter, loc, rewriter.getI1Type(), op.getLock())
                        .getResult();
    auto innerIf =
        mlir::scf::IfOp::create(rewriter, loc, acquired, /*withElseRegion=*/true);

    rewriter.setInsertionPoint(innerIf.thenBlock()->getTerminator());
    cloneCombiningLockBody(op, payload, rewriter);
    SyncCombiningLockReleaseOp::create(rewriter, loc, op.getLock());

    rewriter.setInsertionPoint(innerIf.elseBlock()->getTerminator());
    emitCombiningLockSlowPath(op, payload, captures, wrapper, moduleOp, rewriter);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

void populateConvertSyncToSTDPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<MutexInitLowering, RawMutexLockLowering, RawMutexUnlockLowering,
               MutexCriticalSectionLowering,
               CombiningLockCriticalSectionLowering>(
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
