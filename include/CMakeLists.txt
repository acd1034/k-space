# Add include folders to the library and targets that consume it the SYSTEM
# keyword suppresses warnings for users of the library
if(KSPC_STANDALONE_PROJECT)
  target_include_directories(kspc INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
else()
  target_include_directories(kspc SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
endif()

# Add (header only) external libraries
# find_package(Boost 1.61.0 REQUIRED)
# target_link_libraries(kspc INTERFACE Boost::boost)

find_package(BLAS REQUIRED)
target_link_libraries(kspc INTERFACE BLAS)

find_package(LAPACK REQUIRED)
target_link_libraries(kspc INTERFACE LAPACK)
