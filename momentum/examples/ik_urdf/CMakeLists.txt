cmake_minimum_required(VERSION 3.16.3)

project(momentum)

find_package(rerun_sdk REQUIRED)
find_package(momentum REQUIRED)
find_package(Eigen3 REQUIRED)

link_directories(/usr/local/lib)
file(GLOB MOMENTUM_LIBS /usr/local/lib/libmomentum_*.so)



set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH "/usr/local/lib")

add_executable(using_diff_error_function using_diff_error_function.cpp)
add_executable(using_character_solver_error_function using_character_solver_error_function.cpp)



set(URDF_SOURCE_DIR ${CMAKE_SOURCE_DIR}/urdf)
set(URDF_DEST_DIR ${CMAKE_BINARY_DIR}/urdf)

file(GLOB_RECURSE URDF_FILES RELATIVE ${URDF_SOURCE_DIR} "${URDF_SOURCE_DIR}/*")

add_custom_target(copy_urdf ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${URDF_DEST_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${URDF_SOURCE_DIR} ${URDF_DEST_DIR}
  COMMENT "Copying URDF files to build directory"
)


target_link_libraries(using_character_solver_error_function PRIVATE 

rerun_sdk
${MOMENTUM_LIBS} 
Eigen3::Eigen)


target_link_libraries( 
using_diff_error_function PRIVATE
rerun_sdk
${MOMENTUM_LIBS} 
Eigen3::Eigen)