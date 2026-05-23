add_executable(litenn_torch_manifest_example main.cpp)
target_compile_features(litenn_torch_manifest_example PRIVATE cxx_std_26)
target_link_libraries(litenn_torch_manifest_example PRIVATE LiteNN)

if(TARGET LiteNNCompiler)
    target_link_libraries(litenn_torch_manifest_example PRIVATE LiteNNCompiler)
    target_compile_definitions(litenn_torch_manifest_example PRIVATE LITENN_ENABLE_MLIR=1)
endif()

add_custom_target(litenn_torch_manifest_example_assets
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/manifest.json"
            "$<TARGET_FILE_DIR:litenn_torch_manifest_example>/manifest.json"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/export_fixture.py"
            "$<TARGET_FILE_DIR:litenn_torch_manifest_example>/export_fixture.py"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/manifest.json" "${CMAKE_CURRENT_SOURCE_DIR}/export_fixture.py"
    VERBATIM
)
add_dependencies(litenn_torch_manifest_example litenn_torch_manifest_example_assets)
