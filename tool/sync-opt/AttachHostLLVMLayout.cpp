#include <memory>
#include <optional>
#include <string>

#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/SubtargetFeature.h>

namespace mlir::sync {
namespace {

struct AttachHostLLVMLayoutPass
    : public mlir::PassWrapper<AttachHostLLVMLayoutPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachHostLLVMLayoutPass)

  llvm::StringRef getArgument() const final {
    return "attach-host-llvm-layout";
  }

  llvm::StringRef getDescription() const final {
    return "Attach the host LLVM triple, data layout, and DLTI spec";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::DLTIDialect, mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();

    mlir::ModuleOp moduleOp = getOperation();
    auto tripleString = llvm::sys::getDefaultTargetTriple();
    llvm::Triple triple(tripleString);

    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
        triple, error);
    if (!target) {
      moduleOp.emitError() << "failed to look up LLVM target for host triple "
                           << tripleString << ": " << error;
      return signalPassFailure();
    }

    llvm::SubtargetFeatures features;
    for (const auto &feature : llvm::sys::getHostCPUFeatures())
      features.AddFeature(feature.getKey(), feature.getValue());

    llvm::TargetOptions options;
    std::unique_ptr<llvm::TargetMachine> targetMachine(
        target->createTargetMachine(triple, llvm::sys::getHostCPUName(),
                                    features.getString(), options,
                                    std::nullopt));
    if (!targetMachine) {
      moduleOp.emitError() << "failed to create LLVM target machine for host "
                           << tripleString;
      return signalPassFailure();
    }

    llvm::DataLayout dataLayout = targetMachine->createDataLayout();
    mlir::Builder builder(&getContext());
    moduleOp->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                      builder.getStringAttr(tripleString));
    moduleOp->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
                      builder.getStringAttr(
                          dataLayout.getStringRepresentation()));

    auto dltiSpec = mlir::translateDataLayout(dataLayout, &getContext());
    if (!dltiSpec) {
      moduleOp.emitError() << "failed to translate host data layout into DLTI";
      return signalPassFailure();
    }
    moduleOp->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, dltiSpec);
  }
};

} // namespace

void registerAttachHostLLVMLayoutPass() {
  mlir::PassRegistration<AttachHostLLVMLayoutPass>();
}

} // namespace mlir::sync
