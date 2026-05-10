if(NOT TARGET LiteNNCompiler)
    message(STATUS "LiteNNCompiler is not available; skipping carrier examples")
    return()
endif()

set(LITENN_CARRIER_SYMBOL_PREFIX "litenn_example_module")
set(LITENN_CARRIER_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
if(WIN32)
    set(LITENN_CARRIER_OBJECT "${LITENN_CARRIER_GENERATED_DIR}/${LITENN_CARRIER_SYMBOL_PREFIX}.obj")
else()
    set(LITENN_CARRIER_OBJECT "${LITENN_CARRIER_GENERATED_DIR}/${LITENN_CARRIER_SYMBOL_PREFIX}.o")
endif()

add_executable(litenn_carrier_emit emit_carrier.cpp)
target_link_libraries(litenn_carrier_emit PRIVATE LiteNNCompiler)
target_compile_features(litenn_carrier_emit PRIVATE cxx_std_26)

add_custom_command(
    OUTPUT "${LITENN_CARRIER_OBJECT}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${LITENN_CARRIER_GENERATED_DIR}"
    COMMAND $<TARGET_FILE:litenn_carrier_emit> "${LITENN_CARRIER_OBJECT}" "${LITENN_CARRIER_SYMBOL_PREFIX}"
    DEPENDS litenn_carrier_emit
    VERBATIM
    COMMENT "Generating LiteNN carrier object example"
)

set_source_files_properties("${LITENN_CARRIER_OBJECT}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

add_library(litenn_carrier_image_static STATIC carrier_anchor.cpp "${LITENN_CARRIER_OBJECT}")
target_compile_features(litenn_carrier_image_static PRIVATE cxx_std_26)

add_library(litenn_carrier_image_shared SHARED carrier_anchor.cpp "${LITENN_CARRIER_OBJECT}")
target_compile_features(litenn_carrier_image_shared PRIVATE cxx_std_26)

if(WIN32)
    set(LITENN_CARRIER_DEF "${CMAKE_CURRENT_BINARY_DIR}/litenn_carrier_exports.def")
    file(GENERATE OUTPUT "${LITENN_CARRIER_DEF}" CONTENT
"EXPORTS
    ${LITENN_CARRIER_SYMBOL_PREFIX}_rodata DATA
    ${LITENN_CARRIER_SYMBOL_PREFIX}_rodata_size DATA
    ${LITENN_CARRIER_SYMBOL_PREFIX}_instructions DATA
    ${LITENN_CARRIER_SYMBOL_PREFIX}_instructions_size DATA
")
    target_sources(litenn_carrier_image_shared PRIVATE "${LITENN_CARRIER_DEF}")
endif()

add_executable(litenn_carrier_static static_loader.cpp)
target_link_libraries(litenn_carrier_static PRIVATE LiteNNCompiler litenn_carrier_image_static)
target_compile_features(litenn_carrier_static PRIVATE cxx_std_26)

add_executable(litenn_carrier_shared_loader shared_loader.cpp)
target_link_libraries(litenn_carrier_shared_loader PRIVATE LiteNNCompiler)
target_compile_features(litenn_carrier_shared_loader PRIVATE cxx_std_26)
target_compile_definitions(litenn_carrier_shared_loader PRIVATE
    LITENN_CARRIER_SHARED_FILE="$<TARGET_FILE_NAME:litenn_carrier_image_shared>"
    LITENN_CARRIER_SYMBOL_PREFIX="${LITENN_CARRIER_SYMBOL_PREFIX}"
)
add_dependencies(litenn_carrier_shared_loader litenn_carrier_image_shared)
if(CMAKE_DL_LIBS)
    target_link_libraries(litenn_carrier_shared_loader PRIVATE ${CMAKE_DL_LIBS})
endif()

add_custom_target(litenn_carrier DEPENDS
    litenn_carrier_emit
    litenn_carrier_image_static
    litenn_carrier_image_shared
    litenn_carrier_static
    litenn_carrier_shared_loader
)