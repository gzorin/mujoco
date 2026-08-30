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

#include "experimental/platform/hal/bourbon_renderer.h"

#include <cstddef>
#include <cstring>
#include <span>

#include <mujoco/mujoco.h>
#include "experimental/platform/hal/graphics_mode.h"
#include "render/bourbon/bourbon_context.h"

namespace mujoco::platform {

BourbonRenderer::BourbonRenderer(void* metal_layer, GraphicsMode gfx_mode)
    : gfx_mode_(gfx_mode),
      context_(std::make_unique<render_bourbon::BourbonContext>(metal_layer)) {
  // Initialize render flags from the MuJoCo defaults (mjRNDSTRING[i][1]).
  for (int i = 0; i < mjNRNDFLAG; ++i) {
    render_flags_[i] = (mjRNDSTRING[i][1][0] == '1') ? 1 : 0;
  }
}

BourbonRenderer::~BourbonRenderer() = default;

void BourbonRenderer::Init(const mjModel* model) { context_->Init(model); }

void BourbonRenderer::Render(const mjModel* model, mjData* data,
                             const mjvPerturb* perturb, mjvCamera* camera,
                             const mjvOption* vis_option, int width, int height,
                             std::span<std::byte> pixels,
                             std::span<mjvGeom> extra_geoms) {
  // Stage 1 will translate the model into the retained scene here. For now the
  // context renders the (empty) bourbon pass graph and draws ImGui on top.
  context_->RenderFrame(model, data, camera, width, height);
}

void BourbonRenderer::RenderToTexture(const mjModel* model, mjData* data,
                                      mjvCamera* camera, int width, int height,
                                      std::byte* output) {
  // Not yet implemented; clear the output buffer so callers get a defined image.
  if (output != nullptr && width > 0 && height > 0) {
    std::memset(output, 0, static_cast<size_t>(width) * height * 3);
  }
}

int BourbonRenderer::UploadImage(int texture_id, const std::byte* pixels,
                                 int width, int height, int bpp) {
  // GUI texture upload is implemented in a later stage.
  return 0;
}

mjtByte* BourbonRenderer::GetRenderFlags() { return render_flags_; }

double BourbonRenderer::GetFps() { return context_->GetFps(); }

}  // namespace mujoco::platform
