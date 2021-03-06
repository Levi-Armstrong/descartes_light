#
# @file descartes_light_macros.cmake
# @brief Common Tesseract CMake Macros
#
# @author Levi Armstrong
# @date October 15, 2019
# @version TODO
# @bug No known bugs
#
# @copyright Copyright (c) 2019, Southwest Research Institute
#
# @par License
# Software License Agreement (Apache License)
# @par
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# @par
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

macro(target_process_export target)
  string(TOUPPER "${target}" EXPORT_PREFIX)

  if (NOT DEFINED ${EXPORT_PREFIX}_LIBRARY_SHARED)
    target_compile_definitions(${target} PRIVATE "${EXPORT_PREFIX}_LIBRARY_SHARED")
  endif()

  if(NOT BUILD_SHARED_LIBS)
      target_compile_definitions(${target} PUBLIC "${EXPORT_PREFIX}_STATIC_LIBRARY")
  endif()
endmacro()

macro(descartes_variables)
  if (NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
  endif()

  if (NOT DEFINED DESCARTES_ENABLE_CLANG_TIDY)
    set(DESCARTES_ENABLE_CLANG_TIDY OFF)
  endif()

  if (NOT DEFINED DESCARTES_ENABLE_TESTING)
    set(DESCARTES_ENABLE_TESTING OFF)
  endif()

  if (NOT DEFINED DESCARTES_ENABLE_RUN_TESTING)
    set(DESCARTES_ENABLE_RUN_TESTING OFF)
  endif()

  if (DESCARTES_ENABLE_TESTING_ALL)
    set(DESCARTES_ENABLE_TESTING ON)
    set(DESCARTES_ENABLE_CLANG_TIDY ON)
  endif()

  set(DESCARTES_COMPILE_DEFINITIONS "")
  if (NOT DESCARTES_ENABLE_TESTING AND NOT DESCARTES_ENABLE_TESTING_ALL)
    set(DESCARTES_CLANG_TIDY_ARGS "-line-filter=[{'name':'ikfast.h','lines':[[9999999,9999999]]}, {'name':'fanuc_m20ia10l_manipulator_ikfast_solver.cpp','lines':[[9999999,9999999]]}, {'name':'.h'}, {'name':'.hpp'}, {'name':'.cpp'}]" ${DEFAULT_CLANG_TIDY_WARNING_ARGS})

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(DESCARTES_COMPILE_OPTIONS -mno-avx -Wall -Wextra -Wconversion -Wsign-conversion -Wno-sign-compare)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      set(DESCARTES_COMPILE_OPTIONS -mno-avx -Wall -Wextra -Winconsistent-missing-override -Wconversion -Wsign-conversion)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      set(DESCARTES_COMPILE_OPTIONS "")
      set(DESCARTES_COMPILE_DEFINITIONS _USE_MATH_DEFINES=ON)
    else()
      message(WARNING "${CMAKE_CXX_COMPILER_ID} Unsupported compiler detected.")
    endif()
  else()
    set(DESCARTES_CLANG_TIDY_ARGS "-line-filter=[{'name':'ikfast.h','lines':[[9999999,9999999]]}, {'name':'fanuc_m20ia10l_manipulator_ikfast_solver.cpp','lines':[[9999999,9999999]]}, {'name':'.h'}, {'name':'.hpp'}, {'name':'.cpp'}]" ${DEFAULT_CLANG_TIDY_ERROR_ARGS})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(DESCARTES_COMPILE_OPTIONS -mno-avx -Werror=all -Werror=extra -Werror=conversion -Werror=sign-conversion -Wno-sign-compare)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      set(DESCARTES_COMPILE_OPTIONS -mno-avx -Werror=all -Werror=extra -Werror=inconsistent-missing-override -Werror=conversion -Werror=sign-conversion)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      set(DESCARTES_COMPILE_OPTIONS "")
      set(DESCARTES_COMPILE_DEFINITIONS _USE_MATH_DEFINES=ON)
    else()
      message(WARNING "${CMAKE_CXX_COMPILER_ID} Unsupported compiler detected.")
    endif()
  endif()

  set(DESCARTES_CXX_VERSION 14)
endmacro()
