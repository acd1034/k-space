# Creates a library kspc which is an interface (header files only)
add_library(kspc INTERFACE)
# If you want to use kspc prefer to link against kspc using this alias target
add_library(kspc::kspc ALIAS kspc)
# Set C++ version
target_compile_features(kspc INTERFACE cxx_std_20)
set_target_properties(kspc PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

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
# target_link_libraries(kspc SYSTEM INTERFACE Boost::boost)

find_package(BLAS REQUIRED)
target_link_libraries(kspc INTERFACE BLAS)

find_package(LAPACK REQUIRED)
target_link_libraries(kspc INTERFACE LAPACK)
