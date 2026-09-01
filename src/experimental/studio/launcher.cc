// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "experimental/studio/launcher.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <string>
#include <string_view>

#include <mujoco/mujoco.h>
#include "experimental/platform/hal/graphics_mode.h"
#include "experimental/platform/resources.h"
#include "experimental/platform/sys_utils.h"
#include "experimental/studio/app.h"

namespace mujoco::studio {

namespace {

// Loads the engine plugins (elasticity, actuator, sensor, sdf) shipped
// alongside the executable. Without this, any model referencing one of them
// fails to load. Probes the bundle layout first, then the flat build-tree
// layout -- the same shape as the asset lookup in platform/resources.cc.
void LoadBundledPlugins() {
  const std::filesystem::path module_dir =
      mujoco::platform::GetModuleDir((void*)&LoadBundledPlugins);
  if (module_dir.empty()) {
    return;
  }

  const std::filesystem::path candidates[] = {
      module_dir.parent_path() / "PlugIns" / "mujoco_plugin",
      module_dir / "mujoco_plugin",
  };

  for (const std::filesystem::path& dir : candidates) {
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
      continue;
    }
    mj_loadAllPluginLibraries(
        dir.string().c_str(),
        +[](const char* filename, int first, int count) {
          std::printf("Plugins registered by library '%s':\n", filename);
          for (int i = first; i < first + count; ++i) {
            std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
          }
        });
    return;
  }
}

}  // namespace

int LaunchStudio(int argc, char** argv, LauncherConfig config) {
  const char* home = std::getenv("HOME");
  const std::string ini_path = std::string(home ? home : ".") + "/.mujoco.ini";

  mujoco::platform::RegisterResourceProviders();
  LoadBundledPlugins();

  if (config.gfx_mode.empty()) {
    const char* display = std::getenv("DISPLAY");
    if (display && strcmp(display, ":20") == 0) {
      config.gfx_mode = "opengl_headless";
    }
  }

  const char* session_type = std::getenv("XDG_SESSION_TYPE");
  const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
  if ((session_type && std::string_view(session_type) == "wayland") ||
      wayland_display) {
    if (config.gfx_mode.empty()) {
      config.gfx_mode = "opengl_headless";
    } else if (config.gfx_mode == "opengl") {
      mju_error(
          "Wayland does not support '%s' graphics mode. "
          "Restart with a different graphics mode, or login using X11.",
          config.gfx_mode.c_str());
    }
  }

  // A Finder-launched .app gets no --gfx, so the default is what a
  // double-click actually starts. Bourbon comes first on macOS: without this,
  // the bundle would silently fall back to the classic OpenGL renderer.
#if defined(MUJOCO_USE_BOURBON)
  const mujoco::platform::GraphicsMode default_mode =
      mujoco::platform::GraphicsMode::BourbonMetal;
#elif defined(MUJOCO_USE_FILAMENT)
  const mujoco::platform::GraphicsMode default_mode =
      mujoco::platform::GraphicsMode::FilamentOpenGl;
#else
  const mujoco::platform::GraphicsMode default_mode =
      mujoco::platform::GraphicsMode::ClassicOpenGl;
#endif
  mujoco::platform::GraphicsMode gfx_mode =
      mujoco::platform::GraphicsModeFromString(config.gfx_mode, default_mode);

  // Use config values if they are set (non-default), otherwise use flags.
  mujoco::studio::App app({
    .width = config.window_width,
    .height = config.window_height,
    .ini_path = ini_path,
    .gfx_mode = gfx_mode,
    .title = config.title,
  });

  if (config.model_file.empty()) {
    app.InitEmptyModel();
  } else {
    app.LoadModelFromFile(config.model_file);
  }

  while (app.Update()) {
    app.BuildGui();
    app.Render();
  }
  return 0;
}

}  // namespace mujoco::studio
