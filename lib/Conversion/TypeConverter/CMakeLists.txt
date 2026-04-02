add_mlir_conversion_library(MLIRSyncTypeConverter
  TypeConverter.cpp

  DEPENDS
    MLIRSyncConversionPassIncGen

  LINK_COMPONENTS
    Core

  LINK_LIBS PUBLIC
    MLIRSync
    MLIRArithAttrToLLVMConversion
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
)
