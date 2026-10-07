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

# Locate the bourbon install prefix from an imported target's on-disk location.
# The resolution is config-agnostic (a Debug-only install sets
# IMPORTED_LOCATION_DEBUG, not _RELEASE) and shared with the .app bundle
# assembly, which needs the same paths to embed these libraries. Bourbon
# installs either plain dylibs or, with BOURBON_BUILD_FRAMEWORKS, .framework
# bundles; both live directly in <prefix>/lib.
mujoco_bourbon_framework_dir(Bourbon::BourbonCore _bourbon_core_framework)
if(_bourbon_core_framework)
  get_filename_component(_bourbon_libdir "${_bourbon_core_framework}" DIRECTORY)
else()
  # The bundle embeds whole frameworks rather than patching loose dylibs.
  if(MUJOCO_STUDIO_MACOS_BUNDLE)
    message(FATAL_ERROR "render_bourbon: MUJOCO_STUDIO_MACOS_BUNDLE requires "
      "Bourbon to be installed as frameworks. Rebuild Bourbon with "
      "-DBOURBON_BUILD_FRAMEWORKS=ON, or configure with "
      "-DMUJOCO_STUDIO_MACOS_BUNDLE=OFF.")
  endif()
  mujoco_bourbon_imported_location(Bourbon::BourbonCore _bourbon_core_loc)
  get_filename_component(_bourbon_libdir "${_bourbon_core_loc}" DIRECTORY)
endif()
get_filename_component(_bourbon_prefix "${_bourbon_libdir}/.." ABSOLUTE)

# This library exchanges Eigen types (Affine3f, Transformation, Array4f, ...)
# with the bourbon dylibs BY VALUE, so it MUST compile against the exact Eigen
# tree bourbon's dylibs were built against. A different Eigen build silently
# changes how fixed-size Eigen types are passed across the ABI boundary (e.g.
# Array4f in FP registers vs a GP register / memory), shifting the pointer
# arguments that follow and corrupting the call.
#
# Bourbon vendors its own Eigen and installs it into its prefix at
# include/eigen3; its dylibs are built against that copy (NOT a system/Homebrew
# Eigen, and NOT MuJoCo's fetched Eigen). Default to the copy in bourbon's
# install prefix, overridable if bourbon was built against an Eigen elsewhere.
set(MUJOCO_BOURBON_EIGEN_DIR "${_bourbon_prefix}/include/eigen3" CACHE PATH
    "Eigen include dir that bourbon's dylibs were compiled against. Must match \
exactly to avoid an Eigen ABI mismatch across the bourbon boundary.")
if(NOT EXISTS "${MUJOCO_BOURBON_EIGEN_DIR}/Eigen/Core")
  message(FATAL_ERROR "render_bourbon: could not find the Eigen tree bourbon "
    "was built against. Expected it in bourbon's install prefix at "
    "${_bourbon_prefix}/include/eigen3. Set MUJOCO_BOURBON_EIGEN_DIR to the "
    "correct location (check bourbon's build compile_commands.json for the "
    "-isystem .../eigen3 path).")
endif()
message(STATUS "render_bourbon: using bourbon-matched Eigen at ${MUJOCO_BOURBON_EIGEN_DIR}")
target_include_directories(render_bourbon BEFORE PRIVATE
  "${MUJOCO_BOURBON_EIGEN_DIR}")

target_compile_features(render_bourbon PUBLIC cxx_std_20)

# Bourbon's DG type system identifies non-trivial value types by the ADDRESS of
# an `inline constexpr` template variable (bourbon::detail::dgtype_tag<T>), and
# type-checks DG connections by comparing those addresses across the DSO
# boundary. MuJoCo builds with hidden visibility (CMAKE_CXX_VISIBILITY_PRESET
# hidden + VISIBILITY_INLINES_HIDDEN), which would give this library its own
# hidden copy of that variable — a different address than bourbon's dylib, so
# connect() would (correctly, from its view) reject every connection as a type
# mismatch. Bourbon and its examples build with default visibility; match that
# here so the weak template instantiations coalesce with bourbon's at load time.
set_target_properties(render_bourbon PROPERTIES
  C_VISIBILITY_PRESET default
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF)

# The visibility preset above is necessary but NOT sufficient once LTO is on.
# MuJoCo enables ThinLTO for non-Debug builds (MUJOCO_ENABLE_LTO=ON), so this
# library compiles to LLVM bitcode (-flto=thin). At the final executable link,
# ThinLTO's whole-program pass sees the dgtype_tag<T> weak symbols referenced
# only from within the executable (bourbon's dylibs carry their own copies) and
# INTERNALIZES them -- they become non-external, so dyld can no longer coalesce
# them with bourbon's dylib copies at load time. That gives them a different
# address than bourbon's, and DGType::compatible() (which compares those tag
# addresses across the DSO boundary) rejects every DG connection as a type
# mismatch -- the same failure the visibility preset was meant to fix, resurfacing
# at the LTO stage. A Debug build (LTO off) hides this bug.
#
# Disabling interprocedural optimization for THIS target keeps the boundary tag
# symbols as real weak-external definitions in a regular object file, which LTO
# then cannot internalize, so they coalesce with bourbon's at load. The rest of
# MuJoCo keeps LTO. (Equivalent global escape hatch: -DMUJOCO_ENABLE_LTO=OFF.)
set_target_properties(render_bourbon PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION OFF)

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

# The bourbon libraries are imported SHARED targets linked as @rpath/... . Add
# their install directory (resolved above) to the runtime search path of any
# executable that links this library (propagated as an INTERFACE link option
# through the static platform library to the studio executable).
#
# Deliberately NOT added for a .app bundle build: an absolute fallback would let
# a bundle that failed to embed one of the frameworks still resolve it from the
# local bourbon install and appear to work on this machine only -- exactly the
# failure the bundle exists to prevent.
if(NOT MUJOCO_STUDIO_MACOS_BUNDLE)
  target_link_options(render_bourbon INTERFACE "-Wl,-rpath,${_bourbon_libdir}")
endif()

add_library(mujoco::render_bourbon ALIAS render_bourbon)
