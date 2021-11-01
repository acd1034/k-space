cmake_minimum_required(VERSION 3.8)
project(kspc CXX)

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
    enable_testing()
    add_subdirectory(tests)
endif()
