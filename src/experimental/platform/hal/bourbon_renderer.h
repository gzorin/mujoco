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

#ifndef MUJOCO_SRC_EXPERIMENTAL_PLATFORM_HAL_BOURBON_RENDERER_H_
#define MUJOCO_SRC_EXPERIMENTAL_PLATFORM_HAL_BOURBON_RENDERER_H_

#include <cstddef>
#include <memory>
#include <span>

#include <mujoco/mujoco.h>
#include "experimental/platform/hal/graphics_mode.h"
#include "experimental/platform/hal/renderer.h"

namespace mujoco::render_bourbon {
class BourbonContext;
}  // namespace mujoco::render_bourbon

namespace mujoco::platform {

// A platform::Renderer that renders MuJoCo models through the bourbon Metal
// renderer. All bourbon interaction is delegated to render_bourbon::BourbonContext
// (a PIMPL), so this translation unit stays free of bourbon and Metal headers.
class BourbonRenderer : public Renderer {
 public:
  // `metal_layer` is the CAMetalLayer* from Window::GetMetalLayer().
  BourbonRenderer(void* metal_layer, GraphicsMode gfx_mode);
  ~BourbonRenderer() override;

  void Init(const mjModel* model) override;

  void Render(const mjModel* model, mjData* data, const mjvPerturb* perturb,
              mjvCamera* camera, const mjvOption* vis_option, int width,
              int height, std::span<std::byte> pixels = {},
              std::span<mjvGeom> extra_geoms = {}) override;

  void RenderToTexture(const mjModel* model, mjData* data, mjvCamera* camera,
                       int width, int height, std::byte* output) override;

  int UploadImage(int texture_id, const std::byte* pixels, int width,
                  int height, int bpp) override;

  mjtByte* GetRenderFlags() override;

  double GetFps() override;

 private:
  GraphicsMode gfx_mode_;
  std::unique_ptr<render_bourbon::BourbonContext> context_;
  mjtByte render_flags_[mjNRNDFLAG] = {};
};

}  // namespace mujoco::platform

#endif  // MUJOCO_SRC_EXPERIMENTAL_PLATFORM_HAL_BOURBON_RENDERER_H_
