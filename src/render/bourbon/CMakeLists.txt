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

# The Bourbon Metal renderer backend. Follows the render_noop pattern (a
# standalone static library exposed via a mujoco:: alias) so that bourbon never
# ends up inside libmujoco.
#
# Containment rule: every bourbon:: call lives under this library. BourbonContext
# is a PIMPL whose public header pulls in no bourbon or Metal headers, so
# consumers (mujoco_platform's bourbon_renderer.cc) compile without bourbon's
# include directories.

add_library(render_bourbon STATIC
  bourbon_context.cc
  bourbon_context.h
  support/imgui_metalcpp.h
  support/imgui_metalcpp.mm
)

target_include_directories(render_bourbon
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_compile_features(render_bourbon PUBLIC cxx_std_20)

target_link_libraries(render_bourbon
  PUBLIC
    mujoco::mujoco
  PRIVATE
    Bourbon::BourbonCore
    Bourbon::BourbonTG
    Bourbon::BourbonSG
    Bourbon::BourbonRenderer
    Bourbon::BourbonMetalCPP
    Bourbon::BourbonMath
    dear_imgui
    dear_imgui_Metal
)

# Runtime path to the Metal toolchain for llair::setPathToTools, discovered by
# the root CMakeLists.txt. Overridable at runtime via the MUJOCO_BOURBON_LLAIR_TOOLS
# environment variable.
target_compile_definitions(render_bourbon PRIVATE
  MUJOCO_BOURBON_LLAIR_TOOLS_PATH="${MUJOCO_METAL_TOOLS_PATH}")

# The bourbon libraries are imported SHARED targets linked as @rpath/... . Add
# their install directory to the runtime search path of any executable that
# links this library (propagated as an INTERFACE link option through the static
# platform library to the studio executable).
get_target_property(_bourbon_core_loc Bourbon::BourbonCore IMPORTED_LOCATION_RELEASE)
if(NOT _bourbon_core_loc)
  get_target_property(_bourbon_core_loc Bourbon::BourbonCore IMPORTED_LOCATION)
endif()
if(_bourbon_core_loc)
  get_filename_component(_bourbon_libdir "${_bourbon_core_loc}" DIRECTORY)
  target_link_options(render_bourbon INTERFACE "-Wl,-rpath,${_bourbon_libdir}")
endif()

add_library(mujoco::render_bourbon ALIAS render_bourbon)
