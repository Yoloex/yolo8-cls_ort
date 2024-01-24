# CMakeList.txt : CMake project for yolo8-cls_ort, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("yolo8-cls_ort")

# Add source to this project's executable.
add_executable (yolo8-cls_ort "yolo8-cls_ort.cpp" "yolo8-cls_ort.h" "lodepng.h" "lodepng.cpp")

set(ONNXRUNTIME_ROOTDIR "${ONNXRUNTIME_ROOTDIR}" CACHE PATH "Path to onnxruntime root directory")

include_directories(${PROJECT_SOURCE_DIR})
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
include_directories("${ONNXRUNTIME_ROOTDIR}/include/onnxruntime")
include_directories("${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")

if(MSVC)
	target_link_libraries(yolo8-cls_ort "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib")
elseif(LINUX)
	target_link_libraries(yolo8-cls_ort "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so")
endif()