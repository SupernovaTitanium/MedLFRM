aux_source_directory(. LINK_SVM_SRC)
add_library(linksvm STATIC ${LINK_SVM_SRC})
set(BOOST_INCLUDE_DIR /scratch/cluster/ianyen/ConvexDictionary/MedLFRM/boost_1_57_0)
include_directories(${BOOST_INCLUDE_DIR})
