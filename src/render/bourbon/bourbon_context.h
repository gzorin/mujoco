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

#ifndef MUJOCO_SRC_RENDER_BOURBON_BOURBON_CONTEXT_H_
#define MUJOCO_SRC_RENDER_BOURBON_BOURBON_CONTEXT_H_

#include <array>
#include <memory>

#include <mujoco/mujoco.h>

namespace mujoco::render_bourbon {

// Owns the bourbon core context, swapchain, and per-frame rendering objects for
// a single Metal window, and encapsulates the bourbon frame ordering.
//
// This is the sole seam between MuJoCo and bourbon: the header exposes only
// MuJoCo/standard types (the implementation is a PIMPL), so translation units
// that include it — notably mujoco_platform's bourbon_renderer.cc — compile
// without bourbon's or Metal's include directories. Every bourbon:: and MTL::
// call lives in bourbon_context.cc.
class BourbonContext {
 public:
  // `metal_layer` is the CAMetalLayer* obtained from the window
  // (Window::GetMetalLayer()). The bourbon device/queue are derived from it.
  explicit BourbonContext(void* metal_layer);
  ~BourbonContext();

  BourbonContext(const BourbonContext&) = delete;
  BourbonContext& operator=(const BourbonContext&) = delete;

  // (Re)builds the retained scene for `model`. Safe to call repeatedly on model
  // reload; drains the GPU and tears down any prior scene first.
  void Init(const mjModel* model);

  // Sets the background clear colour (linear, 0..1).
  void SetClearColor(float r, float g, float b, float a);

  // Renders one frame: updates the camera from `camera`, evaluates the bourbon
  // scene/pass graph into the drawable, draws the current ImGui frame on top,
  // then presents. Owns the ImGui::Render() call. The caller must have built the
  // ImGui frame (ImGui::NewFrame ... widgets) beforehand. `shadow_enabled`
  // (mjRND_SHADOW) toggles shadow casting; enabling it switches the pass graph
  // onto a non-Direct submission mode. `vis_option` may be null; mjVIS_TRANSPARENT
  // is read from it to fade dynamic geoms.
  //
  // `perturb` and `extra_geoms` drive the pooled decorations: a private mjvScene
  // is updated with mjCAT_ALL, `extra_geoms` are appended as mjCAT_DECOR, and
  // every decoration geom is realized through a grow-to-high-water node pool (no
  // per-frame node create/destroy). `perturb` and `extra_geoms` may be null/empty.
  void RenderFrame(const mjModel* model, mjData* data,
                   const mjvPerturb* perturb, mjvCamera* camera,
                   const mjvOption* vis_option, int width, int height,
                   bool shadow_enabled, const mjvGeom* extra_geoms,
                   int num_extra_geoms);

  // Frames-per-second, exponentially smoothed.
  double GetFps() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mujoco::render_bourbon

#endif  // MUJOCO_SRC_RENDER_BOURBON_BOURBON_CONTEXT_H_
