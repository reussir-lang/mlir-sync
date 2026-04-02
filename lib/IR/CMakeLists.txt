get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_dialect_library(MLIRSync
  SyncDialect.cpp
  SyncOps.cpp
  SyncTypes.cpp

  DEPENDS
    SyncOpsIncGen

  LINK_COMPONENTS
    Core

  LINK_LIBS PUBLIC
    ${dialect_libs}
)
