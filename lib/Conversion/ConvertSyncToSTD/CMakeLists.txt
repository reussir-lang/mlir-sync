add_mlir_conversion_library(MLIRSyncConvertSyncToSTD
  ConvertSyncToSTD.cpp

  DEPENDS
    MLIRSyncConversionPassIncGen

  LINK_COMPONENTS
    Core

  LINK_LIBS PUBLIC
    MLIRPtrDialect
    MLIRSCFDialect
    MLIRSync
    MLIRTransforms
)
