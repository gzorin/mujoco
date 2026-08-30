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

#include "render/bourbon/support/imgui_metalcpp.h"

#include <imgui.h>
#include <backends/imgui_impl_metal.h>

#include <Metal/Metal.hpp>

// Bridges metal-cpp pointer types to the Objective-C `id` objects the upstream
// dear_imgui Metal backend expects. metal-cpp objects are toll-free bridged to
// their Objective-C counterparts, so a __bridge cast is sufficient.

bool ImGui_ImplMetalCPP_Init(MTL::Device* device) {
  return ImGui_ImplMetal_Init((__bridge id)device);
}

void ImGui_ImplMetalCPP_Shutdown() { ImGui_ImplMetal_Shutdown(); }

void ImGui_ImplMetalCPP_NewFrame(
    MTL::RenderPassDescriptor* render_pass_descriptor) {
  ImGui_ImplMetal_NewFrame(
      (__bridge MTLRenderPassDescriptor*)render_pass_descriptor);
}

void ImGui_ImplMetalCPP_RenderDrawData(
    ImDrawData* draw_data, MTL::CommandBuffer* command_buffer,
    MTL::RenderCommandEncoder* command_encoder) {
  ImGui_ImplMetal_RenderDrawData(draw_data, (__bridge id)command_buffer,
                                 (__bridge id)command_encoder);
}
