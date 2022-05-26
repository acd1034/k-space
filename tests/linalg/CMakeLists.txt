cmake_minimum_required(VERSION 3.8)
project(linalg_tests CXX)

# ${CMAKE_PROJECT_NAME}: project name of the root CMakeLists.txt
# ${PROJECT_NAME}: project name of the current CMakeLists.txt
add_executable(${PROJECT_NAME}
  linalg.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  kspc_tests_config
  kspc::kspc
  Catch2::Catch2
)

add_test(${PROJECT_NAME} ${PROJECT_NAME})
