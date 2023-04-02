cmake_minimum_required(VERSION 2.8)

project(prewitt)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test_naive_prewitt
               test_naive_prewitt.cc
               naive_prewitt_operator.cc)

target_link_libraries(test_naive_prewitt ${OpenCV_LIBS})
