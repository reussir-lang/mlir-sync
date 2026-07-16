add_executable(sync-types-test SyncTypesTest.cpp)

target_link_libraries(sync-types-test
  PRIVATE
    MLIRSync
    MLIRSyncTypeConverter
    MLIRDLTIDialect
    MLIRLLVMDialect
    MLIRParser
)

add_test(NAME sync-types-test COMMAND sync-types-test)

add_custom_target(check-sync-unit
  COMMAND $<TARGET_FILE:sync-types-test>
  DEPENDS sync-types-test
  VERBATIM
)
