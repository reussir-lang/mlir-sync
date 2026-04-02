add_mlir_conversion_library(MLIRSyncConvertSyncToLLVM
  ConvertSyncToLLVM.cpp

  DEPENDS
    MLIRSyncConversionPassIncGen

  LINK_COMPONENTS
    Core

  LINK_LIBS PUBLIC
    MLIRSync
    MLIRSyncTypeConverter
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRPtrDialect
    MLIRPtrToLLVM
    MLIRTransforms
)
