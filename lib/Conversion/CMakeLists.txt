add_subdirectory(TypeConverter)
add_subdirectory(ConvertSyncToSTD)
add_subdirectory(ConvertSyncToLLVM)

set(SYNC_CONVERSION_AGGREGATE_LIBS
  MLIRSyncTypeConverter
  MLIRSyncConvertSyncToSTD
  MLIRSyncConvertSyncToLLVM
)

add_library(MLIRSyncConversion INTERFACE)
target_link_libraries(MLIRSyncConversion
  INTERFACE
    ${SYNC_CONVERSION_AGGREGATE_LIBS}
)
add_dependencies(MLIRSyncConversion ${SYNC_CONVERSION_AGGREGATE_LIBS})
