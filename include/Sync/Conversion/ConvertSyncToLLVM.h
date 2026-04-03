#pragma once

#ifndef SYNC_CONVERSION_CONVERTSYNCTOLLVM_H
#define SYNC_CONVERSION_CONVERTSYNCTOLLVM_H

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Sync/Conversion/TypeConverter.h"
#include "Sync/IR/SyncOps.h"

namespace mlir::sync {

#define GEN_PASS_DECL_CONVERTSYNCTOLLVMPASS
#include "Sync/Conversion/Passes.h.inc"

void configureConvertSyncToLLVMConversionLegality(
    mlir::ConversionTarget &target);

void populateConvertSyncToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

void registerConvertSyncToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace mlir::sync

#endif // SYNC_CONVERSION_CONVERTSYNCTOLLVM_H
