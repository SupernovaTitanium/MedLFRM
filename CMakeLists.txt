cmake_minimum_required(VERSION 2.8.4)
project(MedLFRM)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_subdirectory(src)


add_executable(MedLFRM src/main.cpp)
set(BOOST_INCLUDE_DIR /home/jiaming/src/boost_1_57_0)


target_link_libraries(MedLFRM linksvm svm_multiclass svmlight utils
/home/jiaming/src/boost_1_57_0/stage/lib/libboost_filesystem.a
/home/jiaming/src/boost_1_57_0/stage/lib/libboost_program_options.a /home/jiaming/src/boost_1_57_0/stage/lib/libboost_system.a)

include_directories(${BOOST_INCLUDE_DIR})
