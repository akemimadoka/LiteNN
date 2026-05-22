add_executable(litenn_safetensors_convert ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_compile_features(litenn_safetensors_convert PRIVATE cxx_std_26)
target_link_libraries(litenn_safetensors_convert PRIVATE LiteNN)
litenn_enable_sanitizers(litenn_safetensors_convert)

install(TARGETS litenn_safetensors_convert
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
