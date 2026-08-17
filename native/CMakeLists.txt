cmake_minimum_required(VERSION 3.16)

project(ppocrv6_rknn_service LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Threads REQUIRED)
find_path(RKNN_API_INCLUDE_DIR rknn_api.h REQUIRED)
find_library(RKNN_RT_LIBRARY rknnrt REQUIRED)

add_library(ppocrv6_rknn_core SHARED
    src/ppocrv6_engine.cc
)

target_include_directories(ppocrv6_rknn_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${RKNN_API_INCLUDE_DIR}
)
target_link_libraries(ppocrv6_rknn_core
    PRIVATE ${OpenCV_LIBS} Threads::Threads
    PUBLIC ${RKNN_RT_LIBRARY}
)
target_compile_options(ppocrv6_rknn_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(ppocrv6_ocr apps/ocr_image.cc)
target_link_libraries(ppocrv6_ocr PRIVATE ppocrv6_rknn_core)
set_target_properties(ppocrv6_ocr PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")

add_executable(ppocrv6_ocr_daemon apps/ocr_daemon.cc)
target_link_libraries(ppocrv6_ocr_daemon PRIVATE ppocrv6_rknn_core)
set_target_properties(ppocrv6_ocr_daemon PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")

install(TARGETS ppocrv6_rknn_core ppocrv6_ocr ppocrv6_ocr_daemon
    LIBRARY DESTINATION bin
    RUNTIME DESTINATION bin
)
