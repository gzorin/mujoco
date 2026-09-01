# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script mode (cmake -P): copy one dylib into a .app and normalise its runtime
# search path.
#
#   cmake -DSRC=<file> -DDST_DIR=<dir> [-DRPATH=<rpath>]
#         [-DINSTALL_NAME_TOOL=<path>] [-DOTOOL=<path>]
#         -P MujocoBundleDylib.cmake
#
# Every existing LC_RPATH is removed before RPATH is added. That matters for the
# engine plugins, which carry an absolute build-tree rpath: dyld searches
# rpaths in order, so leaving it in place would make the bundle on the build
# machine silently load libmujoco from the build tree instead of from
# Contents/Frameworks -- i.e. the one machine where a packaging mistake would be
# caught is the one machine that would not catch it.
#
# Clearing first also makes this idempotent, unlike a bare `-add_rpath`, which
# errors out when the entry is already present.

if(NOT SRC OR NOT DST_DIR)
  message(FATAL_ERROR "MujocoBundleDylib: SRC and DST_DIR are required.")
endif()
if(NOT INSTALL_NAME_TOOL)
  set(INSTALL_NAME_TOOL install_name_tool)
endif()
if(NOT OTOOL)
  set(OTOOL otool)
endif()

file(COPY "${SRC}" DESTINATION "${DST_DIR}")

get_filename_component(_name "${SRC}" NAME)
set(_dst "${DST_DIR}/${_name}")

execute_process(
  COMMAND "${OTOOL}" -l "${_dst}"
  OUTPUT_VARIABLE _load_commands
  RESULT_VARIABLE _otool_result)
if(NOT _otool_result EQUAL 0)
  message(FATAL_ERROR "MujocoBundleDylib: otool -l failed on ${_dst}.")
endif()

# LC_RPATH is the only load command rendered as `path <x> (offset <n>)`;
# LC_ID_DYLIB and friends use `name <x> (offset <n>)`.
string(REGEX MATCHALL "path [^\n]+ \\(offset [0-9]+\\)" _rpath_lines
       "${_load_commands}")
foreach(_line IN LISTS _rpath_lines)
  string(REGEX REPLACE "^path (.+) \\(offset [0-9]+\\)$" "\\1" _old "${_line}")
  execute_process(
    COMMAND "${INSTALL_NAME_TOOL}" -delete_rpath "${_old}" "${_dst}"
    RESULT_VARIABLE _delete_result
    OUTPUT_QUIET ERROR_QUIET)
  if(NOT _delete_result EQUAL 0)
    message(FATAL_ERROR
      "MujocoBundleDylib: failed to remove rpath '${_old}' from ${_dst}.")
  endif()
endforeach()

if(RPATH)
  execute_process(
    COMMAND "${INSTALL_NAME_TOOL}" -add_rpath "${RPATH}" "${_dst}"
    RESULT_VARIABLE _add_result)
  if(NOT _add_result EQUAL 0)
    message(FATAL_ERROR
      "MujocoBundleDylib: failed to add rpath '${RPATH}' to ${_dst}.")
  endif()
endif()
