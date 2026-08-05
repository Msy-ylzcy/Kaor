cmake_minimum_required(VERSION 3.20)
project(kaor_ocr_native LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(OpenCV REQUIRED COMPONENTS core)

Python3_add_library(_kaor_ocr_native MODULE WITH_SOABI kaor_ocr_native.cpp)
target_compile_features(_kaor_ocr_native PRIVATE cxx_std_17)
target_include_directories(_kaor_ocr_native PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(_kaor_ocr_native PRIVATE ${OpenCV_LIBS})

set(KAOR_NATIVE_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/../backend")
set_target_properties(
    _kaor_ocr_native
    PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${KAOR_NATIVE_OUTPUT}"
        RUNTIME_OUTPUT_DIRECTORY "${KAOR_NATIVE_OUTPUT}"
)

foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(
        _kaor_ocr_native
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${configuration} "${KAOR_NATIVE_OUTPUT}"
            RUNTIME_OUTPUT_DIRECTORY_${configuration} "${KAOR_NATIVE_OUTPUT}"
    )
endforeach()
