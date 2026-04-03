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
constexpr llvm::StringLiteral kReadLockSlowPath =
    "mlir_sync_rwlock_read_lock_slow_path";
constexpr llvm::StringLiteral kWriteLockSlowPath =
    "mlir_sync_rwlock_write_lock_slow_path";
constexpr llvm::StringLiteral kRwLockUnlockSlowPath =
    "mlir_sync_rwlock_unlock_slow_path";
constexpr llvm::StringLiteral kCombiningAttachSlowPath =
    "mlir_sync_combining_lock_attach_slow_path";
constexpr llvm::StringLiteral kCombiningWrapperPrefix =
    "__sync_combining_lock_slow_";
constexpr uint32_t kRwLockMask = (1u << 30) - 1;
constexpr uint32_t kRwLockReadersWaiting = 1u << 30;
constexpr uint32_t kRwLockWritersWaiting = 1u << 31;

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

mlir::MemRefType getRawRwLockProjectionType(mlir::MemRefType rwlockType) {
  return mlir::MemRefType::get(
      {}, RawRwLockType::get(rwlockType.getContext()),
      mlir::MemRefLayoutAttrInterface{}, rwlockType.getMemorySpace());
}

mlir::MemRefType getRwLockPayloadProjectionType(mlir::MemRefType rwlockType) {
  auto rwlockElementType = llvm::cast<RwLockType>(rwlockType.getElementType());
  return mlir::MemRefType::get({}, rwlockElementType.getValueType(),
                               mlir::MemRefLayoutAttrInterface{},
                               rwlockType.getMemorySpace());
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

mlir::Value createI32Constant(mlir::Location loc, uint32_t value,
                              mlir::OpBuilder &builder) {
  auto type = builder.getI32Type();
  auto attr = mlir::IntegerAttr::get(type, llvm::APInt(32, value));
  return mlir::arith::ConstantOp::create(builder, loc, type, attr);
}

mlir::Value createI1Constant(mlir::Location loc, bool value,
                             mlir::OpBuilder &builder) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value ? 1 : 0, 1)
      .getResult();
}

void emitSlowPathIfFalse(
    mlir::Location loc, mlir::Value condition,
    llvm::function_ref<void()> emitSlowPath, mlir::PatternRewriter &rewriter) {
  auto ifOp = mlir::scf::IfOp::create(rewriter, loc, condition,
                                      /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
  emitSlowPath();
}

void emitSlowPathIfTrue(
    mlir::Location loc, mlir::Value condition,
    llvm::function_ref<void()> emitSlowPath, mlir::PatternRewriter &rewriter) {
  auto ifOp = mlir::scf::IfOp::create(rewriter, loc, condition,
                                      /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  emitSlowPath();
}

mlir::Value buildReadUnlockSlowPathCondition(mlir::Location loc,
                                             mlir::Value state,
                                             mlir::PatternRewriter &rewriter) {
  auto mask = createI32Constant(loc, kRwLockMask, rewriter);
  auto zero = createI32Constant(loc, 0, rewriter);
  auto writersWaiting = createI32Constant(loc, kRwLockWritersWaiting, rewriter);
  auto masked = rewriter.create<mlir::arith::AndIOp>(loc, state, mask);
  auto writers = rewriter.create<mlir::arith::AndIOp>(loc, state, writersWaiting);
  auto unlocked = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, masked, zero);
  auto hasWritersWaiting = rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, writers, zero);
  return rewriter.create<mlir::arith::AndIOp>(loc, unlocked, hasWritersWaiting);
}

mlir::Value buildWriteUnlockSlowPathCondition(mlir::Location loc,
                                              mlir::Value state,
                                              mlir::PatternRewriter &rewriter) {
  auto waiters = createI32Constant(
      loc, kRwLockReadersWaiting | kRwLockWritersWaiting, rewriter);
  auto zero = createI32Constant(loc, 0, rewriter);
  auto pending = rewriter.create<mlir::arith::AndIOp>(loc, state, waiters);
  return rewriter.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, pending, zero);
}

template <typename CriticalSectionOp, typename LockOp, typename UnlockOp,
          typename GetRawOp, typename GetPayloadOp>
mlir::LogicalResult lowerPayloadCriticalSection(
    CriticalSectionOp op, mlir::Value lockValue, mlir::MemRefType rawLockType,
    mlir::MemRefType payloadType, mlir::PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  auto rawLock = GetRawOp::create(rewriter, loc, rawLockType, lockValue).getResult();
  LockOp::create(rewriter, loc, rawLock);
  auto payload =
      GetPayloadOp::create(rewriter, loc, payloadType, lockValue).getResult();

  auto &bodyBlock = op.getBody().front();
  auto yieldOp = llvm::cast<SyncYieldOp>(bodyBlock.getTerminator());
  llvm::SmallVector<mlir::Value> yieldedValues;
  yieldedValues.reserve(yieldOp->getNumOperands());
  for (mlir::Value value : yieldOp->getOperands())
    yieldedValues.push_back(value == bodyBlock.getArgument(0) ? payload : value);
  rewriter.eraseOp(yieldOp);
  rewriter.inlineBlockBefore(&bodyBlock, op, mlir::ValueRange{payload});
  rewriter.setInsertionPoint(op);
  UnlockOp::create(rewriter, loc, rawLock);

  if (op.getNumResults() == 0)
    rewriter.eraseOp(op);
  else
    rewriter.replaceOp(op, yieldedValues);
  return mlir::success();
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
    emitSlowPathIfFalse(
        loc, acquired,
        [&]() {
          auto ptr = materializeRuntimePtr(loc, op.getMutex(), ptrType, rewriter);
          mlir::func::CallOp::create(rewriter, loc, slowPathFunc,
                                     mlir::ValueRange{ptr});
        },
        rewriter);

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

struct RwLockInitLowering : public mlir::OpRewritePattern<SyncRwLockInitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRwLockInitOp op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto rwlockType = llvm::cast<mlir::MemRefType>(op.getRwlock().getType());
    auto rawRwLockType = getRawRwLockProjectionType(rwlockType);
    auto rawRwLock =
        SyncRwLockGetRawRwLockOp::create(rewriter, loc, rawRwLockType, op.getRwlock())
            .getResult();
    SyncRawRwLockInitOp::create(rewriter, loc, rawRwLock);
    if (mlir::Value initialValue = op.getInitialValue()) {
      auto payloadType = getRwLockPayloadProjectionType(rwlockType);
      auto payload =
          SyncRwLockGetPayloadOp::create(rewriter, loc, payloadType, op.getRwlock())
              .getResult();
      mlir::memref::StoreOp::create(rewriter, loc, initialValue, payload,
                                    mlir::ValueRange{});
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RawRwLockTryReadLockLowering
    : public mlir::OpRewritePattern<SyncRawRwLockTryReadLockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRawRwLockTryReadLockOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto falseValue = createI1Constant(loc, false, rewriter);
    auto trueValue = createI1Constant(loc, true, rewriter);
    auto whileOp = mlir::scf::WhileOp::create(
        rewriter, loc, mlir::TypeRange{rewriter.getI1Type(), rewriter.getI1Type()},
        mlir::ValueRange{falseValue, falseValue},
        [&](mlir::OpBuilder &builder, mlir::Location beforeLoc, mlir::ValueRange args) {
          auto done = args[0];
          auto shouldContinue = mlir::arith::CmpIOp::create(
              builder, beforeLoc, mlir::arith::CmpIPredicate::eq, done,
              createI1Constant(beforeLoc, false, rewriter));
          mlir::scf::ConditionOp::create(builder, beforeLoc,
                                         shouldContinue.getResult(), args);
        },
        [&](mlir::OpBuilder &builder, mlir::Location afterLoc, mlir::ValueRange) {
          auto state = SyncRawRwLockLoadStateOp::create(builder, afterLoc,
                                                        builder.getI32Type(),
                                                        op.getRwlock())
                           .getResult();
          auto mask = createI32Constant(afterLoc, kRwLockMask, rewriter);
          auto waiters = createI32Constant(
              afterLoc, kRwLockReadersWaiting | kRwLockWritersWaiting, rewriter);
          auto maxReaders = createI32Constant(afterLoc, kRwLockMask - 1, rewriter);
          auto zero = createI32Constant(afterLoc, 0, rewriter);
          auto readLocked = createI32Constant(afterLoc, 1, rewriter);
          auto masked = mlir::arith::AndIOp::create(builder, afterLoc, state, mask);
          auto waitersMasked =
              mlir::arith::AndIOp::create(builder, afterLoc, state, waiters);
          auto hasCapacity = mlir::arith::CmpIOp::create(
              builder, afterLoc, mlir::arith::CmpIPredicate::ult, masked,
              maxReaders);
          auto noWaiters = mlir::arith::CmpIOp::create(
              builder, afterLoc, mlir::arith::CmpIPredicate::eq, waitersMasked,
              zero);
          auto readLockable = mlir::arith::AndIOp::create(builder, afterLoc,
                                                          hasCapacity, noWaiters);

          auto ifOp = mlir::scf::IfOp::create(
              builder, afterLoc,
              mlir::TypeRange{builder.getI1Type(), builder.getI1Type()},
              readLockable, /*withElseRegion=*/true);
          builder.setInsertionPointToStart(ifOp.thenBlock());
          auto updated =
              mlir::arith::AddIOp::create(builder, afterLoc, state, readLocked);
          auto acquired = SyncRawRwLockCmpxchgStateOp::create(
                              builder, afterLoc, builder.getI1Type(), op.getRwlock(),
                              state, updated.getResult())
                              .getResult();
          mlir::scf::YieldOp::create(builder, afterLoc,
                                     mlir::ValueRange{acquired, acquired});

          builder.setInsertionPointToStart(ifOp.elseBlock());
          mlir::scf::YieldOp::create(builder, afterLoc,
                                     mlir::ValueRange{trueValue, falseValue});

          builder.setInsertionPointAfter(ifOp);
          mlir::scf::YieldOp::create(builder, afterLoc, ifOp.getResults());
        });
    rewriter.replaceOp(op, whileOp.getResult(1));
    return mlir::success();
  }
};

struct RawRwLockTryWriteLockLowering
    : public mlir::OpRewritePattern<SyncRawRwLockTryWriteLockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRawRwLockTryWriteLockOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto falseValue = createI1Constant(loc, false, rewriter);
    auto trueValue = createI1Constant(loc, true, rewriter);
    auto whileOp = mlir::scf::WhileOp::create(
        rewriter, loc, mlir::TypeRange{rewriter.getI1Type(), rewriter.getI1Type()},
        mlir::ValueRange{falseValue, falseValue},
        [&](mlir::OpBuilder &builder, mlir::Location beforeLoc, mlir::ValueRange args) {
          auto done = args[0];
          auto shouldContinue = mlir::arith::CmpIOp::create(
              builder, beforeLoc, mlir::arith::CmpIPredicate::eq, done,
              createI1Constant(beforeLoc, false, rewriter));
          mlir::scf::ConditionOp::create(builder, beforeLoc,
                                         shouldContinue.getResult(), args);
        },
        [&](mlir::OpBuilder &builder, mlir::Location afterLoc, mlir::ValueRange) {
          auto state = SyncRawRwLockLoadStateOp::create(builder, afterLoc,
                                                        builder.getI32Type(),
                                                        op.getRwlock())
                           .getResult();
          auto zero = createI32Constant(afterLoc, 0, rewriter);
          auto writeLocked = createI32Constant(afterLoc, kRwLockMask, rewriter);
          auto isUnlocked = mlir::arith::CmpIOp::create(
              builder, afterLoc, mlir::arith::CmpIPredicate::eq, state, zero);

          auto ifOp = mlir::scf::IfOp::create(
              builder, afterLoc,
              mlir::TypeRange{builder.getI1Type(), builder.getI1Type()},
              isUnlocked, /*withElseRegion=*/true);
          builder.setInsertionPointToStart(ifOp.thenBlock());
          auto acquired = SyncRawRwLockCmpxchgStateOp::create(
                              builder, afterLoc, builder.getI1Type(), op.getRwlock(),
                              zero, writeLocked)
                              .getResult();
          mlir::scf::YieldOp::create(builder, afterLoc,
                                     mlir::ValueRange{acquired, acquired});

          builder.setInsertionPointToStart(ifOp.elseBlock());
          mlir::scf::YieldOp::create(builder, afterLoc,
                                     mlir::ValueRange{trueValue, falseValue});

          builder.setInsertionPointAfter(ifOp);
          mlir::scf::YieldOp::create(builder, afterLoc, ifOp.getResults());
        });
    rewriter.replaceOp(op, whileOp.getResult(1));
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
    emitSlowPathIfTrue(
        loc, needsWake,
        [&]() {
          auto slowPathPtr =
              materializeRuntimePtr(loc, op.getMutex(), ptrType, rewriter);
          mlir::func::CallOp::create(rewriter, loc, unlockSlowFunc,
                                     mlir::ValueRange{slowPathPtr});
        },
        rewriter);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RawRwLockReadLockLowering
    : public mlir::OpRewritePattern<SyncRawRwLockReadLockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRawRwLockReadLockOp op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto ptrType = getRuntimePtrType(op.getRwlock());
    auto slowPathFunc = getOrCreateRuntimeFunc(loc, kReadLockSlowPath, {ptrType},
                                               {}, moduleOp, rewriter);
    auto acquired = SyncRawRwLockTryReadLockOp::create(rewriter, loc,
                                                       rewriter.getI1Type(),
                                                       op.getRwlock())
                        .getResult();
    emitSlowPathIfFalse(
        loc, acquired,
        [&]() {
          auto ptr = materializeRuntimePtr(loc, op.getRwlock(), ptrType, rewriter);
          mlir::func::CallOp::create(rewriter, loc, slowPathFunc,
                                     mlir::ValueRange{ptr});
        },
        rewriter);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RawRwLockReadUnlockLowering
    : public mlir::OpRewritePattern<SyncRawRwLockReadUnlockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRawRwLockReadUnlockOp op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto ptrType = getRuntimePtrType(op.getRwlock());
    auto unlockSlowFunc = getOrCreateRuntimeFunc(
        loc, kRwLockUnlockSlowPath, {ptrType, rewriter.getI32Type()}, {}, moduleOp,
        rewriter);
    auto state = SyncRawRwLockReadUnlockFastOp::create(rewriter, loc,
                                                       rewriter.getI32Type(),
                                                       op.getRwlock())
                     .getResult();
    auto needsWake = buildReadUnlockSlowPathCondition(loc, state, rewriter);
    emitSlowPathIfTrue(
        loc, needsWake,
        [&]() {
          auto ptr = materializeRuntimePtr(loc, op.getRwlock(), ptrType, rewriter);
          mlir::func::CallOp::create(rewriter, loc, unlockSlowFunc,
                                     mlir::ValueRange{ptr, state});
        },
        rewriter);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RawRwLockWriteLockLowering
    : public mlir::OpRewritePattern<SyncRawRwLockWriteLockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRawRwLockWriteLockOp op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto ptrType = getRuntimePtrType(op.getRwlock());
    auto slowPathFunc = getOrCreateRuntimeFunc(loc, kWriteLockSlowPath, {ptrType},
                                               {}, moduleOp, rewriter);
    auto acquired = SyncRawRwLockTryWriteLockOp::create(rewriter, loc,
                                                        rewriter.getI1Type(),
                                                        op.getRwlock())
                        .getResult();
    emitSlowPathIfFalse(
        loc, acquired,
        [&]() {
          auto ptr = materializeRuntimePtr(loc, op.getRwlock(), ptrType, rewriter);
          mlir::func::CallOp::create(rewriter, loc, slowPathFunc,
                                     mlir::ValueRange{ptr});
        },
        rewriter);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct RawRwLockWriteUnlockLowering
    : public mlir::OpRewritePattern<SyncRawRwLockWriteUnlockOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRawRwLockWriteUnlockOp op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    auto ptrType = getRuntimePtrType(op.getRwlock());
    auto unlockSlowFunc = getOrCreateRuntimeFunc(
        loc, kRwLockUnlockSlowPath, {ptrType, rewriter.getI32Type()}, {}, moduleOp,
        rewriter);
    auto state = SyncRawRwLockWriteUnlockFastOp::create(rewriter, loc,
                                                        rewriter.getI32Type(),
                                                        op.getRwlock())
                     .getResult();
    auto needsWake = buildWriteUnlockSlowPathCondition(loc, state, rewriter);
    emitSlowPathIfTrue(
        loc, needsWake,
        [&]() {
          auto ptr = materializeRuntimePtr(loc, op.getRwlock(), ptrType, rewriter);
          mlir::func::CallOp::create(rewriter, loc, unlockSlowFunc,
                                     mlir::ValueRange{ptr, state});
        },
        rewriter);
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
    auto mutexType = llvm::cast<mlir::MemRefType>(op.getMutex().getType());
    auto rawMutexType = getRawMutexProjectionType(mutexType);
    auto payloadType = getPayloadProjectionType(mutexType);
    return lowerPayloadCriticalSection<SyncMutexCriticalSectionOp,
                                       SyncRawMutexLockOp, SyncRawMutexUnlockOp,
                                       SyncMutexGetRawMutexOp,
                                       SyncMutexGetPayloadOp>(
        op, op.getMutex(), rawMutexType, payloadType, rewriter);
  }
};

struct RwLockReadCriticalSectionLowering
    : public mlir::OpRewritePattern<SyncRwLockReadCriticalSectionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRwLockReadCriticalSectionOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto rwlockType = llvm::cast<mlir::MemRefType>(op.getRwlock().getType());
    auto rawRwLockType = getRawRwLockProjectionType(rwlockType);
    auto payloadType = getRwLockPayloadProjectionType(rwlockType);
    return lowerPayloadCriticalSection<SyncRwLockReadCriticalSectionOp,
                                       SyncRawRwLockReadLockOp,
                                       SyncRawRwLockReadUnlockOp,
                                       SyncRwLockGetRawRwLockOp,
                                       SyncRwLockGetPayloadOp>(
        op, op.getRwlock(), rawRwLockType, payloadType, rewriter);
  }
};

struct RwLockWriteCriticalSectionLowering
    : public mlir::OpRewritePattern<SyncRwLockWriteCriticalSectionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SyncRwLockWriteCriticalSectionOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto rwlockType = llvm::cast<mlir::MemRefType>(op.getRwlock().getType());
    auto rawRwLockType = getRawRwLockProjectionType(rwlockType);
    auto payloadType = getRwLockPayloadProjectionType(rwlockType);
    return lowerPayloadCriticalSection<SyncRwLockWriteCriticalSectionOp,
                                       SyncRawRwLockWriteLockOp,
                                       SyncRawRwLockWriteUnlockOp,
                                       SyncRwLockGetRawRwLockOp,
                                       SyncRwLockGetPayloadOp>(
        op, op.getRwlock(), rawRwLockType, payloadType, rewriter);
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
    auto trueValue =
        mlir::arith::ConstantIntOp::create(rewriter, loc, 1, 1).getResult();
    auto falseValue =
        mlir::arith::ConstantIntOp::create(rewriter, loc, 0, 1).getResult();
    auto hasTail = SyncCombiningLockHasTailOp::create(
                       rewriter, loc, rewriter.getI1Type(), op.getLock())
                       .getResult();
    auto shouldSlowPath = mlir::scf::IfOp::create(
        rewriter, loc, mlir::TypeRange{rewriter.getI1Type()}, hasTail,
        /*withElseRegion=*/true);

    rewriter.setInsertionPointToStart(shouldSlowPath.thenBlock());
    mlir::scf::YieldOp::create(rewriter, loc, trueValue);

    rewriter.setInsertionPointToStart(shouldSlowPath.elseBlock());
    auto acquired = SyncCombiningLockTryAcquireOp::create(
                        rewriter, loc, rewriter.getI1Type(), op.getLock())
                        .getResult();
    auto innerIf = mlir::scf::IfOp::create(
        rewriter, loc, mlir::TypeRange{rewriter.getI1Type()}, acquired,
        /*withElseRegion=*/true);

    rewriter.setInsertionPointToStart(innerIf.thenBlock());
    mlir::scf::YieldOp::create(rewriter, loc, falseValue);

    rewriter.setInsertionPointToStart(innerIf.elseBlock());
    mlir::scf::YieldOp::create(rewriter, loc, trueValue);

    rewriter.setInsertionPointAfter(innerIf);
    mlir::scf::YieldOp::create(rewriter, loc, innerIf.getResult(0));

    rewriter.setInsertionPointAfter(shouldSlowPath);
    auto dispatchIf = mlir::scf::IfOp::create(
        rewriter, loc, shouldSlowPath.getResult(0), /*withElseRegion=*/true);

    rewriter.setInsertionPoint(dispatchIf.thenBlock()->getTerminator());
    emitCombiningLockSlowPath(op, payload, captures, wrapper, moduleOp, rewriter);

    rewriter.setInsertionPoint(dispatchIf.elseBlock()->getTerminator());
    cloneCombiningLockBody(op, payload, rewriter);
    SyncCombiningLockReleaseOp::create(rewriter, loc, op.getLock());

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

void populateConvertSyncToSTDPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<MutexInitLowering, RwLockInitLowering, RawMutexLockLowering,
               RawMutexUnlockLowering, RawRwLockTryReadLockLowering,
               RawRwLockTryWriteLockLowering, RawRwLockReadLockLowering,
               RawRwLockReadUnlockLowering, RawRwLockWriteLockLowering,
               RawRwLockWriteUnlockLowering, MutexCriticalSectionLowering,
               RwLockReadCriticalSectionLowering,
               RwLockWriteCriticalSectionLowering,
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
