# Copyright 2025 DeepMind Technologies Limited
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

cmake_minimum_required(VERSION 3.16)

include(third_party_deps/dear_imgui)
include(third_party_deps/implot)
include(third_party_deps/atkinson_hyperlegible_next)
include(third_party_deps/atkinson_hyperlegible_mono)
include(third_party_deps/font_awesome)

# Common configuration shared between mujoco_studio and mujoco_live.
function(configure_studio_target TARGET_NAME)
    target_sources(${TARGET_NAME}
        PRIVATE
            app.cc
            app.h
    )

    target_include_directories(${TARGET_NAME}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
    )

    if (WIN32)
        target_compile_definitions(${TARGET_NAME}
            PRIVATE
            -D_USE_MATH_DEFINES
        )
        set_target_properties(${TARGET_NAME}
            PROPERTIES
                VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        )
    endif()

    target_link_libraries(${TARGET_NAME}
        PRIVATE
            absl::flags
            absl::flags_parse
            dear_imgui
            implot
            mujoco::mujoco
            mujoco::platform
    )

    # Self-registering plugin TUs (mjPLUGIN_LIB_INIT) have no referenced symbols,
    # so the linker discards them when they arrive inside a static archive.
    if(APPLE)
        target_link_options(${TARGET_NAME} PRIVATE
            "-Wl,-force_load,$<TARGET_FILE:mujoco_platform>")
    elseif(NOT MSVC)
        target_link_options(${TARGET_NAME} PRIVATE
            "-Wl,--whole-archive" "$<TARGET_FILE:mujoco_platform>" "-Wl,--no-whole-archive")
    else()
        target_link_options(${TARGET_NAME} PRIVATE
            "/WHOLEARCHIVE:$<TARGET_FILE:mujoco_platform>")
    endif()

    get_target_property(IS_BUNDLE ${TARGET_NAME} MACOSX_BUNDLE)
    if(IS_BUNDLE)
        set(STUDIO_ASSETS_DIR $<TARGET_FILE_DIR:${TARGET_NAME}>/../Resources/assets)
    else()
        set(STUDIO_ASSETS_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets)
    endif()

    add_custom_command(
        TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${STUDIO_ASSETS_DIR}
        COMMAND ${CMAKE_COMMAND}
            -E copy
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../_deps/atkinson_hyperlegible_next-src/fonts/variable/AtkinsonHyperlegibleNext[wght].ttf"
                ${STUDIO_ASSETS_DIR}/AtkinsonHyperlegibleNext[wght].ttf
        COMMENT "Copying AtkinsonHyperlegibleNext font to build directory"
    )
    add_custom_command(
        TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${STUDIO_ASSETS_DIR}
        COMMAND ${CMAKE_COMMAND}
            -E copy
                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../_deps/font_awesome-src/fonts/fontawesome-webfont.ttf
                ${STUDIO_ASSETS_DIR}
        COMMENT "Copying fontawesome-webfont.ttf to build directory"
    )
    add_custom_command(
        TARGET ${TARGET_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${STUDIO_ASSETS_DIR}
        COMMAND ${CMAKE_COMMAND}
            -E copy
                "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../_deps/atkinson_hyperlegible_mono-src/fonts/ttf/AtkinsonHyperlegibleMono-Regular.ttf"
                ${STUDIO_ASSETS_DIR}/AtkinsonHyperlegibleMono-Regular.ttf
        COMMENT "Copying AtkinsonHyperlegibleMono-Regular font to build directory"
    )

    # Filament backend requires additional files to be copied into an "assets" folder.
    if(MUJOCO_USE_FILAMENT)
        add_custom_command(
            TARGET ${TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${STUDIO_ASSETS_DIR}
            COMMAND ${CMAKE_COMMAND}
                -E copy_directory
                    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/../src/render/filament/assets
                    ${STUDIO_ASSETS_DIR}
            COMMENT "Copying Filament assets to build directory"
        )
    endif()
endfunction()

# Desktop Studio target.
if(NOT EMSCRIPTEN)
    if(APPLE)
        set(STUDIO_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../../dist/mujoco_studio.icns)
    elseif(WIN32)
        set(STUDIO_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../../dist/mujoco_studio.rc)
    else()
        set(STUDIO_RESOURCE_FILES "")
    endif()

    add_executable(mujoco_studio
        main.cc
        launcher.cc
        launcher.h
        ${STUDIO_RESOURCE_FILES}
    )

    if(APPLE AND MUJOCO_STUDIO_MACOS_BUNDLE)
        # --- Info.plist substitution values -------------------------------------
        #
        # Git state is captured at configure time, so it goes stale until the next
        # re-configure. That is the right trade for a dev-facing app: these keys
        # exist to make a bug report unambiguous, not to gate anything.
        execute_process(
            COMMAND git rev-parse --short HEAD
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_VARIABLE MUJOCO_STUDIO_GIT_REVISION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(NOT MUJOCO_STUDIO_GIT_REVISION)
            set(MUJOCO_STUDIO_GIT_REVISION "unknown")
        else()
            execute_process(
                COMMAND git status --porcelain --untracked-files=no
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE _studio_git_dirty
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(_studio_git_dirty)
                set(MUJOCO_STUDIO_GIT_REVISION "${MUJOCO_STUDIO_GIT_REVISION}-dirty")
            endif()
        endif()

        # CFBundleVersion must be monotonic -- that is what macOS compares for
        # launch-services and update purposes -- so it is derived from the commit
        # count rather than from the SHA.
        execute_process(
            COMMAND git rev-list --count HEAD
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_VARIABLE MUJOCO_STUDIO_BUILD_NUMBER
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(NOT MUJOCO_STUDIO_BUILD_NUMBER)
            set(MUJOCO_STUDIO_BUILD_NUMBER "0")
        endif()

        set(MUJOCO_STUDIO_BOURBON_REVISION "none")
        if(MUJOCO_USE_BOURBON)
            file(READ "${PROJECT_SOURCE_DIR}/src/render/bourbon/BOURBON_VERSION"
                 _bourbon_version_text)
            if(_bourbon_version_text MATCHES "bourbon git SHA: ([0-9a-f]+)")
                set(MUJOCO_STUDIO_BOURBON_REVISION "${CMAKE_MATCH_1}")
            endif()
        endif()

        set(MUJOCO_STUDIO_RENDER_BACKENDS "classic")
        if(MUJOCO_USE_FILAMENT)
            set(MUJOCO_STUDIO_RENDER_BACKENDS
                "${MUJOCO_STUDIO_RENDER_BACKENDS}, filament")
        endif()
        if(MUJOCO_USE_BOURBON)
            set(MUJOCO_STUDIO_RENDER_BACKENDS
                "${MUJOCO_STUDIO_RENDER_BACKENDS}, bourbon")
        endif()

        # CMAKE_OSX_DEPLOYMENT_TARGET is normally unset here: MujocoMacOS.cmake
        # passes -mmacosx-version-min directly so as not to affect SDK choice.
        set(MUJOCO_STUDIO_MIN_MACOS "${CMAKE_OSX_DEPLOYMENT_TARGET}")
        if(NOT MUJOCO_STUDIO_MIN_MACOS)
            set(MUJOCO_STUDIO_MIN_MACOS "${MUJOCO_MACOSX_VERSION_MIN}.0")
        endif()

        # --- bundle properties ---------------------------------------------------
        #
        # MACOSX_BUNDLE must be set BEFORE configure_studio_target() below, which
        # reads the property to decide whether fonts are staged into
        # Contents/Resources/assets or into bin/assets.
        set(_studio_install_rpath "@executable_path/../Frameworks")
        if(MUJOCO_BUILD_MACOS_FRAMEWORKS)
            # Keep the framework build's extra entry; the two options are orthogonal.
            list(APPEND _studio_install_rpath "@executable_path/../../..")
        endif()

        set_target_properties(mujoco_studio PROPERTIES
            MACOSX_BUNDLE TRUE
            OUTPUT_NAME "MuJoCo Studio"
            MACOSX_BUNDLE_INFO_PLIST "${PROJECT_SOURCE_DIR}/dist/Info.plist.studio.in"
            MACOSX_BUNDLE_BUNDLE_NAME "MuJoCo Studio"
            MACOSX_BUNDLE_GUI_IDENTIFIER "org.mujoco.studio"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
            MACOSX_BUNDLE_BUNDLE_VERSION
                "${PROJECT_VERSION}.${MUJOCO_STUDIO_BUILD_NUMBER}"
            MACOSX_BUNDLE_LONG_VERSION_STRING
                "${PROJECT_VERSION} (${MUJOCO_STUDIO_GIT_REVISION})"
            MACOSX_BUNDLE_INFO_STRING
                "${PROJECT_VERSION} (${MUJOCO_STUDIO_GIT_REVISION})"
            MACOSX_BUNDLE_ICON_FILE "mujoco_studio.icns"
            MACOSX_BUNDLE_COPYRIGHT
                "Copyright 2026 DeepMind Technologies Limited."
            RESOURCE "${STUDIO_RESOURCE_FILES}"
            INSTALL_RPATH "${_studio_install_rpath}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    elseif(APPLE AND MUJOCO_BUILD_MACOS_FRAMEWORKS)
        # Bundling was explicitly disabled, so fall back to the bare bundle the
        # framework layout has always produced (nothing is embedded into it).
        set_target_properties(mujoco_studio PROPERTIES
            MACOSX_BUNDLE TRUE
            INSTALL_RPATH "@executable_path/../Frameworks;@executable_path/../../.."
            BUILD_WITH_INSTALL_RPATH TRUE
            RESOURCE "${STUDIO_RESOURCE_FILES}"
            MACOSX_BUNDLE_ICON_FILE "mujoco_studio.icns"
        )
    endif()
    if(WIN32)
        add_custom_command(
            TARGET mujoco_studio
            PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../../../dist/mujoco_studio.ico
                    ${CMAKE_CURRENT_SOURCE_DIR}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_CURRENT_SOURCE_DIR}/mujoco_studio.ico
        )
    endif()
    configure_studio_target(mujoco_studio)

    # Stage libmujoco, the bourbon dylibs and the engine plugins into the .app,
    # then re-sign it. POST_BUILD (not install-time) so the bundle launched out
    # of the build tree is the same artifact that ships.
    if(APPLE AND MUJOCO_STUDIO_MACOS_BUNDLE)
        mujoco_studio_assemble_bundle(mujoco_studio)
        mujoco_studio_add_dmg_target(mujoco_studio)
    endif()

    # A bundle manages its own rpath (BUILD_WITH_INSTALL_RPATH above); appending
    # the install lib dir would reintroduce an absolute path into the .app.
    if(NOT (APPLE AND (MUJOCO_BUILD_MACOS_FRAMEWORKS OR MUJOCO_STUDIO_MACOS_BUNDLE)))
        include(TargetAddRpath)
        target_add_rpath(
          TARGETS
          mujoco_studio
          INSTALL_DIRECTORY
          "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
          LIB_DIRS
          "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
          DEPENDS
          MUJOCO_ENABLE_RPATH
        )
    endif()
    install(
      TARGETS mujoco_studio
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT studio
      BUNDLE DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT studio
    )
    get_target_property(IS_BUNDLE mujoco_studio MACOSX_BUNDLE)
    if(NOT IS_BUNDLE)
        install(
          DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets"
          DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT studio
        )
    endif()
endif()

# WASM Live target.
if(EMSCRIPTEN)
    add_executable(mujoco_studio)
    target_sources(mujoco_studio PRIVATE emscripten.cc)
    target_compile_options(mujoco_studio PRIVATE -g)
    configure_studio_target(mujoco_studio)
    target_link_libraries(mujoco_studio PRIVATE mujoco::render_noop)

    # Ensure resource decoder plugins are linked into the WASM binary.
    target_link_options(mujoco_studio PRIVATE
        -Wl,--whole-archive $<TARGET_FILE:mujoco> -Wl,--no-whole-archive
    )

    # Filament's OpenGL backend requires WebGL2 / full ES3 bindings.
    target_link_options(mujoco_studio PRIVATE
        --bind
        -sUSE_WEBGL2=1
        -sFULL_ES3
        -sMIN_WEBGL_VERSION=2
        -sMAX_WEBGL_VERSION=2
        -sALLOW_MEMORY_GROWTH=1
        -sASYNCIFY=1
        -sFETCH=1
        -sGL_PREINITIALIZED_CONTEXT=1
        -sSTACK_SIZE=32mb
        -sINITIAL_MEMORY=1024mb
        -sMAXIMUM_MEMORY=4gb
        -sASSERTIONS=1
        -fexceptions
        -g
    )
endif()
