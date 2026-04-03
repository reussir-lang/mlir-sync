#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "Sync/Conversion/ConvertSyncToLLVM.h"
#include "Sync/Conversion/Passes.h"
#include "Sync/IR/SyncDialect.h"

namespace mlir::sync {
void registerAttachHostLLVMLayoutPass();
} // namespace mlir::sync

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::sync::SyncDialect>();
  mlir::sync::registerConvertSyncToLLVMInterface(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::sync::createConvertSyncToSTDPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::sync::createConvertSyncToLLVMPass();
  });
  mlir::sync::registerAttachHostLLVMLayoutPass();

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "sync analysis and optimization driver\n", registry));
}
