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

#include "render/bourbon/bourbon_context.h"

#include <chrono>
#include <memory>
#include <utility>

#include <imgui.h>

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <BourbonCore/CommandBuffer.h>
#include <BourbonCore/CommandQueue.h>
#include <BourbonCore/CoreContext.h>
#include <BourbonCore/Drawable.h>
#include <BourbonCore/Framebuffer.h>
#include <BourbonCore/RenderCommandEncoder.h>
#include <BourbonCore/RenderPass.h>
#include <BourbonCore/Swapchain.h>

#include <mujoco/mujoco.h>

#include "render/bourbon/support/imgui_metalcpp.h"

namespace mujoco::render_bourbon {

struct BourbonContext::Impl {
  // Owned Metal objects. The device is owned by the CAMetalLayer (SDL created
  // it); we only own the command queue we allocate against it.
  CA::MetalLayer* metal_layer = nullptr;
  MTL::Device* device = nullptr;
  NS::SharedPtr<MTL::CommandQueue> mtl_queue;

  // Bourbon core objects (single window, single frame in flight).
  std::unique_ptr<bourbon::CoreContext> core;
  std::unique_ptr<bourbon::Swapchain> swapchain;
  bourbon::CommandQueue* queue = nullptr;  // owned by core.

  // A reusable framebuffer whose single colour attachment is rebound to each
  // frame's drawable; the render pass clears it before ImGui draws on top.
  std::unique_ptr<bourbon::Framebuffer> ui_framebuffer;

  const mjModel* model = nullptr;

  MTL::ClearColor clear_color = MTL::ClearColor(0.12, 0.14, 0.18, 1.0);

  // FPS tracking.
  std::chrono::steady_clock::time_point last_frame;
  bool have_last_frame = false;
  double fps = 0.0;

  // Drains all in-flight GPU work: a no-op command buffer submitted and waited
  // on. Required before tearing down heaps or rebuilding the pass graph.
  void DrainGpu() {
    if (!core || !queue) return;
    auto fence = bourbon::CommandBuffer::Create(*queue, *core);
    fence->enqueue();
    fence->submit();
    fence->metal_command_buffer()->waitUntilCompleted();
  }
};

BourbonContext::BourbonContext(void* metal_layer)
    : impl_(std::make_unique<Impl>()) {
  if (!metal_layer) {
    mju_error("BourbonContext: null Metal layer");
  }
  impl_->metal_layer = reinterpret_cast<CA::MetalLayer*>(metal_layer);
  impl_->device = impl_->metal_layer->device();
  if (!impl_->device) {
    mju_error("BourbonContext: CAMetalLayer has no MTLDevice");
  }

  impl_->mtl_queue = NS::TransferPtr(impl_->device->newCommandQueue());
  impl_->core = std::make_unique<bourbon::CoreContext>(impl_->device,
                                                       impl_->mtl_queue.get());
  impl_->swapchain =
      bourbon::Swapchain::Create(impl_->metal_layer, *impl_->core);
  impl_->queue = impl_->core->getCommandQueue();

  // A render pass that clears colour attachment 0 and stores the result, plus a
  // framebuffer to hold the (per-frame) drawable image.
  auto* attachment = bourbon::RenderPassAttachment::Get(
      impl_->swapchain->format(), MTL::LoadActionClear, MTL::StoreActionStore,
      *impl_->core);
  auto* render_pass = bourbon::RenderPass::Get(
      bourbon::RenderPassColorAttachments::AttachmentsVector{attachment},
      *impl_->core);
  impl_->ui_framebuffer =
      bourbon::Framebuffer::Create(render_pass, *impl_->core);
  impl_->ui_framebuffer->setColorClearValue(0, impl_->clear_color);

  ImGui_ImplMetalCPP_Init(impl_->device);
}

BourbonContext::~BourbonContext() {
  impl_->DrainGpu();
  ImGui_ImplMetalCPP_Shutdown();
}

void BourbonContext::Init(const mjModel* model) {
  // Stage 1 will build the retained scene here. For now only track the model.
  impl_->DrainGpu();
  impl_->model = model;
}

void BourbonContext::SetClearColor(float r, float g, float b, float a) {
  impl_->clear_color = MTL::ClearColor(r, g, b, a);
  if (impl_->ui_framebuffer) {
    impl_->ui_framebuffer->setColorClearValue(0, impl_->clear_color);
  }
}

void BourbonContext::RenderFrame() {
  Impl& s = *impl_;

  // Must be first each frame: lets released heap blocks be stamped against the
  // frame that is about to complete.
  s.core->advanceCompletionTimestamp();

  std::unique_ptr<bourbon::Drawable> drawable = s.swapchain->getNextDrawable();
  if (!drawable) {
    // No drawable available this frame (e.g. window occluded); still finalize
    // the ImGui frame so its internal state stays consistent.
    ImGui::EndFrame();
    return;
  }

  s.ui_framebuffer->setColorAttachment(0, drawable->image());
  s.ui_framebuffer->setColorClearValue(0, s.clear_color);

  ImGui_ImplMetalCPP_NewFrame(s.ui_framebuffer->metal_render_pass_descriptor());
  ImGui::Render();

  auto commands = bourbon::CommandBuffer::Create(*s.queue, *s.core);
  auto encoder =
      bourbon::RenderCommandEncoder::Create(*commands, *s.ui_framebuffer);
  ImGui_ImplMetalCPP_RenderDrawData(ImGui::GetDrawData(),
                                    commands->metal_command_buffer(),
                                    encoder->metal_render_command_encoder());
  encoder->end();

  commands->signalCompletion();
  commands->present(std::shared_ptr<bourbon::Drawable>(std::move(drawable)));
  commands->submit();

  // Update smoothed FPS.
  auto now = std::chrono::steady_clock::now();
  if (s.have_last_frame) {
    double dt = std::chrono::duration<double>(now - s.last_frame).count();
    if (dt > 0.0) {
      double inst = 1.0 / dt;
      s.fps = s.fps > 0.0 ? 0.9 * s.fps + 0.1 * inst : inst;
    }
  }
  s.last_frame = now;
  s.have_last_frame = true;
}

double BourbonContext::GetFps() const { return impl_->fps; }

}  // namespace mujoco::render_bourbon
