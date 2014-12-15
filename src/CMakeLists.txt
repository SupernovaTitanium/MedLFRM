add_subdirectory(LinkSVM)
add_subdirectory(SVMLight)
add_subdirectory(SVM_Multiclass)
add_subdirectory(utils)

set(BOOST_INCLUDE_DIR "/home/jiaming/src/boost_1_57_0")
set(BOOST_LIBRARY_DIR "/home/jiaming/src/boost_1_57_0/stage/lib")

include_directories("${BOOST_INCLUDE_DIR}")
link_directories("${BOOST_LIBRARY_DIR}")

add_executable(MedLFRM main.cpp)
target_link_libraries(MedLFRM linksvm svm_multiclass svmlight utils libboost_filesystem.a libboost_program_options.a libboost_system.a)
