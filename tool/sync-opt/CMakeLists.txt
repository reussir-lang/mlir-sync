get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_tool(sync-opt
  AttachHostLLVMLayout.cpp
  main.cpp
  DEPENDS
    MLIROptLib
    MLIRSync
    MLIRSyncConversion
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
)

target_link_libraries(sync-opt
  PRIVATE
    MLIRSync
    MLIRSyncConversion
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
)

if(LLVM_VERSION VERSION_GREATER_EQUAL 22)
  target_link_libraries(sync-opt
    PRIVATE
      MLIRRegisterAllDialects
      MLIRRegisterAllExtensions
      MLIRRegisterAllPasses
  )
endif()
