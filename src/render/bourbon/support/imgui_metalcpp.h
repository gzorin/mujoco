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

// A thin metal-cpp shim over dear_imgui's Objective-C++ Metal backend
// (backends/imgui_impl_metal.mm), so the renderer can drive ImGui using
// metal-cpp pointer types without touching Objective-C directly. Adapted from
// bourbon's examples/imgui/imgui_impl_metalcpp.{h,mm}.

#ifndef MUJOCO_SRC_RENDER_BOURBON_SUPPORT_IMGUI_METALCPP_H_
#define MUJOCO_SRC_RENDER_BOURBON_SUPPORT_IMGUI_METALCPP_H_

#include <imgui.h>  // IMGUI_IMPL_API, ImDrawData

namespace MTL {
class CommandBuffer;
class Device;
class RenderCommandEncoder;
class RenderPassDescriptor;
}  // namespace MTL

IMGUI_IMPL_API bool ImGui_ImplMetalCPP_Init(MTL::Device*);
IMGUI_IMPL_API void ImGui_ImplMetalCPP_Shutdown();
IMGUI_IMPL_API void ImGui_ImplMetalCPP_NewFrame(MTL::RenderPassDescriptor*);
IMGUI_IMPL_API void ImGui_ImplMetalCPP_RenderDrawData(ImDrawData*,
                                                      MTL::CommandBuffer*,
                                                      MTL::RenderCommandEncoder*);

#endif  // MUJOCO_SRC_RENDER_BOURBON_SUPPORT_IMGUI_METALCPP_H_
