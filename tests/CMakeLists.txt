cmake_minimum_required(VERSION 3.8)
project(kspc_tests CXX)
include(CTest)

# This interface adds compile options to how the tests are run
add_library(kspc_tests_config INTERFACE)
target_compile_options(kspc_tests_config INTERFACE
  -fno-strict-aliasing
  -Wall
  -Wcast-align
  -Wconversion
  -Wctor-dtor-privacy
  # -Werror
  -Wextra
  -Wpedantic
  -Wshadow
  -Wsign-conversion
  # -Wno-deprecated-declarations # Allow tests for [[deprecated]] elements
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
    -Weverything
    -Wno-c++98-compat
    -Wno-c++98-compat-pedantic
    -Wno-missing-braces
    -Wno-covered-switch-default # GTest
    -Wno-global-constructors # GTest
    -Wno-missing-prototypes
    -Wno-padded
    -Wno-unknown-attributes
    -Wno-used-but-marked-unused # GTest EXPECT_DEATH
    -Wno-weak-vtables
  >
  # $<$<CXX_COMPILER_ID:Clang>:
  #   $<$<AND:$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,4.99>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,6>>:
  #     $<$<EQUAL:${KSPC_CXX_STANDARD},17>:-Wno-undefined-func-template>
  #   >
  # >
  # $<$<CXX_COMPILER_ID:AppleClang>:
  #   $<$<AND:$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,9.1>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,10>>:
  #     $<$<EQUAL:${KSPC_CXX_STANDARD},17>:-Wno-undefined-func-template>
  #   >
  # >
  $<$<CXX_COMPILER_ID:GNU>:
    -Wdouble-promotion # float implicit to double
    -Wlogical-op # suspicious uses of logical operators
    $<$<NOT:$<VERSION_LESS:$<CXX_COMPILER_VERSION>,6>>:
      -Wduplicated-cond # duplicated if-else conditions
      -Wmisleading-indentation
      -Wnull-dereference
    >
    $<$<NOT:$<VERSION_LESS:$<CXX_COMPILER_VERSION>,7>>:
      -Wduplicated-branches # identical if-else branches
    >
  >
)

# Import the globally installed kspc if CMake has been started independently in
# this directory with tests
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  find_package(kspc::kspc REQUIRED)
endif()

# Use Catch2 as a unit testing framework
include(FetchContent)
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.6)
FetchContent_MakeAvailable(Catch2)

add_subdirectory(linalg)
add_subdirectory(math)
add_subdirectory(ranges)
