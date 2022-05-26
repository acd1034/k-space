cmake_minimum_required(VERSION 3.8)
project(kspc CXX)
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

# Specify build type if not
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
  message(STATUS "CMAKE_BUILD_TYPE not specified: Use Release by default.")
endif(NOT CMAKE_BUILD_TYPE)

# Determine whether this is a standalone project or included by other projects
set(KSPC_STANDALONE_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(KSPC_STANDALONE_PROJECT ON)
endif()

# Project options
option(KSPC_TEST "Build and perform kspc tests" ${KSPC_STANDALONE_PROJECT})

# Setup include directory
add_subdirectory(include)

# Setup test directory
if (KSPC_TEST)
  include(CTest)
  add_subdirectory(tests)
endif()
