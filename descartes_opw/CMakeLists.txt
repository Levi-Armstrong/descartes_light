cmake_minimum_required(VERSION 3.5.0)

# Extract package name and version
find_package(cmake_common_scripts REQUIRED)
extract_package_metadata(pkg)
project(${pkg_extracted_name} VERSION ${pkg_extracted_version} LANGUAGES CXX)

find_package(descartes_light REQUIRED)
find_package(opw_kinematics REQUIRED)
find_package(console_bridge REQUIRED)

if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(NOT TARGET console_bridge::console_bridge)
  add_library(console_bridge::console_bridge INTERFACE IMPORTED)
  set_target_properties(console_bridge::console_bridge PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${console_bridge_INCLUDE_DIRS})
  set_target_properties(console_bridge::console_bridge PROPERTIES INTERFACE_LINK_LIBRARIES ${console_bridge_LIBRARIES})
else()
  get_target_property(CHECK_INCLUDE_DIRECTORIES console_bridge::console_bridge INTERFACE_INCLUDE_DIRECTORIES)
  if (NOT ${CHECK_INCLUDE_DIRECTORIES})
    set_target_properties(console_bridge::console_bridge PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${console_bridge_INCLUDE_DIRS})
  endif()
endif()

descartes_variables()

if(NOT MSVC)
  initialize_code_coverage()
  add_code_coverage_all_targets(ENABLE ${DESCARTES_ENABLE_TESTING})
endif()

# Declare a C++ library
add_library(${PROJECT_NAME} src/descartes_opw_kinematics.cpp)
target_process_export(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PUBLIC console_bridge::console_bridge opw_kinematics::opw_kinematics descartes::descartes_light)
target_cxx_version(${PROJECT_NAME} PUBLIC VERSION ${DESCARTES_CXX_VERSION})
target_compile_options(${PROJECT_NAME} PUBLIC ${DESCARTES_COMPILE_OPTIONS})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${DESCARTES_COMPILE_DEFINITIONS})
if(NOT MSVC)
  target_clang_tidy(${PROJECT_NAME} ENABLE ${DESCARTES_ENABLE_CLANG_TIDY} ARGUMENTS ${DESCARTES_CLANG_TIDY_ARGS})
  target_code_coverage(${PROJECT_NAME} ALL ENABLE ${DESCARTES_ENABLE_TESTING})
endif()
target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC)

configure_package(NAMESPACE descartes TARGETS ${PROJECT_NAME})

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
  PATTERN ".svn" EXCLUDE
 )
