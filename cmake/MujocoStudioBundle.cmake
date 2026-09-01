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

# Assembly of the relocatable "MuJoCo Studio.app" bundle (macOS only).
#
# Everything here runs POST_BUILD rather than at install time, deliberately: the
# .app that gets launched day to day out of the build tree must be the same
# artifact that ships, otherwise a missing dylib only shows up after packaging.

if(NOT APPLE)
  return()
endif()

set(MUJOCO_STUDIO_CODESIGN_IDENTITY "-" CACHE STRING
    "Codesign identity for the MuJoCo Studio bundle. '-' means ad-hoc. \
install_name_tool invalidates the linker's ad-hoc signature, and an invalid \
signature is a hard load failure on Apple Silicon, so re-signing is mandatory.")

find_program(MUJOCO_INSTALL_NAME_TOOL install_name_tool)
find_program(MUJOCO_OTOOL otool)
find_program(MUJOCO_CODESIGN codesign)
mark_as_advanced(MUJOCO_INSTALL_NAME_TOOL MUJOCO_OTOOL MUJOCO_CODESIGN)

set(MUJOCO_BUNDLE_DYLIB_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/MujocoBundleDylib.cmake")

# The five bourbon dylibs the studio executable resolves through @rpath.
set(MUJOCO_BOURBON_RUNTIME_TARGETS
    Bourbon::BourbonCore
    Bourbon::BourbonTG
    Bourbon::BourbonSG
    Bourbon::BourbonRenderer
    Bourbon::BourbonMetalCPP)

# Resolves the on-disk location of an imported target, config-agnostically: a
# Debug-only install sets IMPORTED_LOCATION_DEBUG and no IMPORTED_LOCATION, so
# walking IMPORTED_CONFIGURATIONS first is what makes this work regardless of
# how the dependency was built.
function(mujoco_bourbon_imported_location IMPORTED_TARGET OUT_VAR)
  if(NOT TARGET ${IMPORTED_TARGET})
    message(FATAL_ERROR
      "mujoco_bourbon_imported_location: no such target ${IMPORTED_TARGET}.")
  endif()
  get_target_property(_cfgs ${IMPORTED_TARGET} IMPORTED_CONFIGURATIONS)
  set(_loc "")
  if(_cfgs)
    list(GET _cfgs 0 _cfg0)
    get_target_property(_loc ${IMPORTED_TARGET} IMPORTED_LOCATION_${_cfg0})
  endif()
  if(NOT _loc)
    get_target_property(_loc ${IMPORTED_TARGET} IMPORTED_LOCATION)
  endif()
  if(NOT _loc)
    message(FATAL_ERROR
      "mujoco_bourbon_imported_location: could not resolve ${IMPORTED_TARGET} "
      "imported location (configs: ${_cfgs}).")
  endif()
  set(${OUT_VAR} "${_loc}" PARENT_SCOPE)
endfunction()

# Appends a "copy SRC into DST_DIR and reset its LC_RPATHs to RPATH" step to the
# command list held in LIST_VAR. See cmake/MujocoBundleDylib.cmake for why the
# existing rpaths are cleared rather than appended to.
function(_mujoco_studio_append_embed_dylib LIST_VAR SRC DST_DIR RPATH)
  set(_cmds ${${LIST_VAR}})
  list(APPEND _cmds
       COMMAND "${CMAKE_COMMAND}"
               "-DSRC=${SRC}"
               "-DDST_DIR=${DST_DIR}"
               "-DRPATH=${RPATH}"
               "-DINSTALL_NAME_TOOL=${MUJOCO_INSTALL_NAME_TOOL}"
               "-DOTOOL=${MUJOCO_OTOOL}"
               -P "${MUJOCO_BUNDLE_DYLIB_SCRIPT}")
  set(${LIST_VAR} ${_cmds} PARENT_SCOPE)
endfunction()

# Appends a `codesign --force` of PATH to the command list held in LIST_VAR.
# Signing happens inside-out (nested code first, bundle last) because the
# top-level signature seals whatever the nested items look like at that moment.
function(_mujoco_studio_append_codesign LIST_VAR PATH)
  set(_cmds ${${LIST_VAR}})
  list(APPEND _cmds
       COMMAND "${MUJOCO_CODESIGN}" --force --sign
               "${MUJOCO_STUDIO_CODESIGN_IDENTITY}" --timestamp=none "${PATH}")
  set(${LIST_VAR} ${_cmds} PARENT_SCOPE)
endfunction()

# Stages libmujoco, the bourbon dylibs and the engine plugins into TARGET_NAME's
# .app, rewrites their runtime search paths, and re-signs the result.
function(mujoco_studio_assemble_bundle TARGET_NAME)
  if(NOT MUJOCO_INSTALL_NAME_TOOL OR NOT MUJOCO_OTOOL)
    message(FATAL_ERROR
      "mujoco_studio_assemble_bundle: install_name_tool and otool are required.")
  endif()
  if(NOT MUJOCO_CODESIGN)
    message(FATAL_ERROR "mujoco_studio_assemble_bundle: codesign not found.")
  endif()

  set(_contents "$<TARGET_BUNDLE_CONTENT_DIR:${TARGET_NAME}>")
  set(_frameworks "${_contents}/Frameworks")
  set(_plugins "${_contents}/PlugIns/mujoco_plugin")

  set(_cmds COMMAND mkdir -p "${_frameworks}")
  set(_sign_cmds "")

  # --- libmujoco -------------------------------------------------------------
  if(MUJOCO_BUILD_MACOS_FRAMEWORKS)
    # libmujoco is laid out as mujoco.framework; copy the whole bundle, mirroring
    # simulate/CMakeLists.txt. The symlink and the .tbd are dropped because they
    # cannot be signed.
    list(APPEND _cmds
         COMMAND rm -rf "${_frameworks}/mujoco.framework"
         COMMAND cp -a "$<TARGET_FILE_DIR:mujoco>/../../../mujoco.framework"
                 "${_frameworks}/"
         COMMAND rm -rf "${_frameworks}/mujoco.framework/mujoco.tbd"
         COMMAND rm -rf
                 "${_frameworks}/mujoco.framework/Versions/A/libmujoco.dylib")
    _mujoco_studio_append_codesign(_sign_cmds "${_frameworks}/mujoco.framework")
  else()
    get_target_property(_mujoco_type mujoco TYPE)
    if(_mujoco_type STREQUAL "SHARED_LIBRARY")
      # TARGET_SONAME_FILE, not TARGET_FILE: the executable links against
      # @rpath/libmujoco.<version>.dylib, and libmujoco.dylib is only a symlink.
      # No rpath of its own: it has no @rpath dependencies.
      _mujoco_studio_append_embed_dylib(
        _cmds "$<TARGET_SONAME_FILE:mujoco>" "${_frameworks}" "")
      _mujoco_studio_append_codesign(
        _sign_cmds "${_frameworks}/$<TARGET_SONAME_FILE_NAME:mujoco>")
    endif()
  endif()

  # --- bourbon dylibs --------------------------------------------------------
  if(MUJOCO_USE_BOURBON)
    foreach(_bourbon_target IN LISTS MUJOCO_BOURBON_RUNTIME_TARGETS)
      mujoco_bourbon_imported_location(${_bourbon_target} _bourbon_loc)
      get_filename_component(_bourbon_name "${_bourbon_loc}" NAME)
      # These dylibs reference each other as @rpath/libBourbon*.dylib. Resolving
      # that through the executable's own LC_RPATH happens to work for a normal
      # link, but is not guaranteed once anything dlopens them, and @loader_path
      # costs nothing.
      _mujoco_studio_append_embed_dylib(
        _cmds "${_bourbon_loc}" "${_frameworks}" "@loader_path")
      _mujoco_studio_append_codesign(_sign_cmds "${_frameworks}/${_bourbon_name}")
    endforeach()
  endif()

  # --- engine plugins --------------------------------------------------------
  # Studio has never called mj_loadAllPluginLibraries, so models needing these
  # failed to load; launcher.cc now probes Contents/PlugIns/mujoco_plugin.
  set(_plugin_targets elasticity actuator sensor sdf_plugin)
  set(_have_plugins FALSE)
  foreach(_plugin IN LISTS _plugin_targets)
    if(TARGET ${_plugin})
      if(NOT _have_plugins)
        list(APPEND _cmds COMMAND mkdir -p "${_plugins}")
        set(_have_plugins TRUE)
      endif()
      add_dependencies(${TARGET_NAME} ${_plugin})
      # The *copies* are patched, not the plugin targets, so non-bundle builds
      # keep their existing rpaths untouched. The copies drop the build-tree
      # rpath they inherit and resolve @rpath/libmujoco.*.dylib out of the
      # bundle instead.
      _mujoco_studio_append_embed_dylib(
        _cmds "$<TARGET_FILE:${_plugin}>" "${_plugins}"
        "@loader_path/../../Frameworks")
      _mujoco_studio_append_codesign(
        _sign_cmds "${_plugins}/$<TARGET_FILE_NAME:${_plugin}>")
    endif()
  endforeach()

  # --- signing ---------------------------------------------------------------
  # install_name_tool invalidates the ad-hoc signature the linker applied, and
  # on Apple Silicon an invalid signature is a hard load failure. The bundle is
  # signed last so it seals the patched nested code.
  _mujoco_studio_append_codesign(
    _sign_cmds "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>")

  add_custom_command(
    TARGET ${TARGET_NAME}
    POST_BUILD
    ${_cmds}
    ${_sign_cmds}
    COMMENT "Assembling and signing ${TARGET_NAME}.app"
    VERBATIM
  )
endfunction()

# Adds a `mujoco_studio_dmg` target (excluded from `all`) producing a
# drag-to-install disk image of TARGET_NAME's bundle.
function(mujoco_studio_add_dmg_target TARGET_NAME)
  get_target_property(_output_name ${TARGET_NAME} OUTPUT_NAME)
  if(NOT _output_name)
    set(_output_name ${TARGET_NAME})
  endif()

  set(_stage "${CMAKE_BINARY_DIR}/dmg-stage")
  set(_dmg "${CMAKE_BINARY_DIR}/MuJoCo-Studio-${PROJECT_VERSION}.dmg")
  set(_readme "${CMAKE_CURRENT_BINARY_DIR}/dmg-README.txt")

  file(WRITE "${_readme}"
"MuJoCo Studio ${PROJECT_VERSION}\n"
"\n"
"Drag \"${_output_name}.app\" onto the Applications folder to install.\n"
"\n"
"REQUIREMENT: the Xcode Metal toolchain.\n"
"The Metal renderer compiles its shader pipelines at runtime, so the app\n"
"cannot bundle this one dependency. If rendering fails on first launch with a\n"
"message about the Metal toolchain, install it with:\n"
"\n"
"    xcodebuild -downloadComponent MetalToolchain\n"
"\n"
"A machine with only the Command Line Tools installed is not sufficient.\n"
"\n"
"This app is ad-hoc signed, not notarized. macOS will refuse to open it on\n"
"first launch; right-click the app and choose Open, or clear the quarantine\n"
"attribute:\n"
"\n"
"    xattr -dr com.apple.quarantine \"/Applications/${_output_name}.app\"\n")

  add_custom_target(mujoco_studio_dmg
    # ditto rather than `cmake -E copy_directory`: it is the macOS-native copy
    # and preserves the permissions and metadata the signature depends on.
    COMMAND rm -rf "${_stage}"
    COMMAND mkdir -p "${_stage}"
    COMMAND ditto "$<TARGET_BUNDLE_DIR:${TARGET_NAME}>"
            "${_stage}/${_output_name}.app"
    COMMAND ln -s /Applications "${_stage}/Applications"
    COMMAND "${CMAKE_COMMAND}" -E copy "${_readme}" "${_stage}/README.txt"
    COMMAND rm -f "${_dmg}"
    COMMAND hdiutil create -volname "MuJoCo Studio ${PROJECT_VERSION}"
            -srcfolder "${_stage}" -ov -format ULFO "${_dmg}"
    COMMAND rm -rf "${_stage}"
    COMMENT "Building ${_dmg}"
    VERBATIM
  )
  add_dependencies(mujoco_studio_dmg ${TARGET_NAME})
endfunction()
