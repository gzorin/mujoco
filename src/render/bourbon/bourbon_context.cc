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
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <memory_resource>
#include <utility>
#include <vector>

#include <imgui.h>

#include <Eigen/Geometry>

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <BourbonCore/CommandBuffer.h>
#include <BourbonCore/CommandQueue.h>
#include <BourbonCore/CoreContext.h>
#include <BourbonCore/DeviceImage.h>
#include <BourbonCore/Drawable.h>
#include <BourbonCore/Framebuffer.h>
#include <BourbonCore/Future.h>
#include <BourbonCore/PersistentHeap.h>
#include <BourbonCore/RenderCommandEncoder.h>
#include <BourbonCore/RenderPass.h>
#include <BourbonCore/Swapchain.h>

#include <BourbonMath/Matrices.h>
#include <BourbonMath/Matrices.cpp>
#include <BourbonMath/Transformation.h>

#include <BourbonTG/DGSource.h>
#include <BourbonTG/DGDeviceImagePlugs.h>
#include <BourbonTG/TaskGraph.h>
#include <BourbonTG/TaskGraphEvaluator.h>

#include <BourbonSG/Camera.h>
#include <BourbonSG/LightSource.h>
#include <BourbonSG/LightSources/DistantLight.h>
#include <BourbonSG/Material.h>
#include <BourbonSG/MatrixTransformer.h>
#include <BourbonSG/Model.h>
#include <BourbonSG/Pattern.h>
#include <BourbonSG/Projections/MatrixProjection.h>
#include <BourbonSG/SGContext.h>
#include <BourbonSG/SGNode.h>
#include <BourbonSG/ShapeInstance.h>
#include <BourbonSG/Shapes/SDF3D.h>

#include <BourbonCore/Token.h>

#include <BourbonRenderer/RendererContext.h>
#include <BourbonRenderer/World/BXDFIntegrationPipelines.h>
#include <BourbonRenderer/World/DrawSubmission.h>
#include <BourbonRenderer/World/MaterialEvaluationPipelines.h>
#include <BourbonRenderer/World/World.h>
#include <BourbonRenderer/Integrators/Integrator.h>
#include <BourbonRenderer/Integrators/ForwardIntegrator.h>
#include <BourbonRenderer/Passes/ClearFramebufferPass.h>
#include <BourbonRenderer/Passes/ForwardVizPass.h>
#include <BourbonRenderer/Passes/ResolvePass.h>

#include <mujoco/mujoco.h>

#include "render/bourbon/support/imgui_metalcpp.h"

// WORKAROUND: llair::setPathToTools(llvm::StringRef) must run before bourbon
// compiles any Metal pipeline, but the installed bourbon package ships neither
// the llair header nor LLVM's headers. We declare an ABI-compatible minimal
// llvm::StringRef (a {const char*, size_t} standard-layout, trivially-copyable
// pair, matching LLVM's layout and calling convention) so the symbol resolves
// from BourbonCore at link time. Bourbon-side fix: install llair/Tools/Tools.h
// (or expose a const char* overload of setPathToTools).
namespace llvm {
class StringRef {
 public:
  StringRef(const char* s) : Data(s), Length(s ? __builtin_strlen(s) : 0) {}

 private:
  const char* Data;
  __SIZE_TYPE__ Length;
};
}  // namespace llvm
namespace llair {
void setPathToTools(llvm::StringRef path);
}  // namespace llair

namespace mujoco::render_bourbon {

namespace {

using DrawableImageSource =
    bourbon::DGSource<bourbon::SharedFuture<bourbon::DeviceImage<2>>>;

constexpr MTL::PixelFormat kColorFormat = MTL::PixelFormatRGBA16Float;
constexpr MTL::PixelFormat kDepthFormat = MTL::PixelFormatDepth32Float_Stencil8;

// Directory containing the Metal toolchain, baked in by CMake; overridable at
// runtime for machines whose toolchain lives elsewhere.
const char* MetalToolsPath() {
  if (const char* env = std::getenv("MUJOCO_BOURBON_LLAIR_TOOLS")) {
    return env;
  }
  return MUJOCO_BOURBON_LLAIR_TOOLS_PATH;
}

bourbon::DeviceImage<2>::Extent ToExtent(CGSize size) {
  return bourbon::DeviceImage<2>::Extent{static_cast<unsigned>(size.width),
                                         static_cast<unsigned>(size.height)};
}

// A retained scene element: an owned SGNode with a rigid transform, a shared
// shape, and a material. Sizes are baked into the shape's parameters and the
// node transform is kept strictly rigid (rotation + translation), which is
// required for the ray-marched SDF primitives (a node scale would break their
// distance field; see the plan's R2).
struct Renderable {
  bourbon::SGNode* node = nullptr;  // owned by world->model()
  bourbon::RefPtr<bourbon::MatrixTransformer> xform;
  bourbon::RefPtr<bourbon::Shape> shape;
  bourbon::RefPtr<bourbon::Material> material;
};

// Builds a rigid object->world transform from a MuJoCo position (3) and a
// row-major 3x3 orientation (mjData::geom_xmat is row-major). Eigen's comma
// initializer fills row-major, so this loads geom_xmat without transposing.
Eigen::Affine3f GeomAffine(const mjtNum* pos, const mjtNum* xmat) {
  Eigen::Matrix3f r;
  r << static_cast<float>(xmat[0]), static_cast<float>(xmat[1]),
      static_cast<float>(xmat[2]), static_cast<float>(xmat[3]),
      static_cast<float>(xmat[4]), static_cast<float>(xmat[5]),
      static_cast<float>(xmat[6]), static_cast<float>(xmat[7]),
      static_cast<float>(xmat[8]);
  Eigen::Affine3f a = Eigen::Affine3f::Identity();
  a.linear() = r;
  a.translation() = Eigen::Vector3f(static_cast<float>(pos[0]),
                                    static_cast<float>(pos[1]),
                                    static_cast<float>(pos[2]));
  return a;
}

// Resolves the effective RGBA of geom `i`, mirroring MuJoCo's setMaterial()
// (engine_vis_visualize.c:225): the material colour is used when the geom has a
// material, but the per-geom rgba overrides it whenever it differs from the
// default (0.5,0.5,0.5,1) or the geom has no material.
void EffectiveGeomRgba(const mjModel* m, int i, float out[4]) {
  const int matid = m->geom_matid[i];
  const float* geom_rgba = m->geom_rgba + 4 * i;
  if (matid >= 0) {
    const float* mr = m->mat_rgba + 4 * matid;
    out[0] = mr[0]; out[1] = mr[1]; out[2] = mr[2]; out[3] = mr[3];
  }
  if (geom_rgba[0] != 0.5f || geom_rgba[1] != 0.5f || geom_rgba[2] != 0.5f ||
      geom_rgba[3] != 1.0f || matid < 0) {
    out[0] = geom_rgba[0]; out[1] = geom_rgba[1];
    out[2] = geom_rgba[2]; out[3] = geom_rgba[3];
  }
}

// Packs an RGBA colour into a 32-bit key (8 bits/channel) so materials that
// share a flat colour are deduplicated exactly.
uint32_t QuantizeRgba(const float rgba[4]) {
  auto q = [](float v) -> uint32_t {
    const float c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return static_cast<uint32_t>(c * 255.0f + 0.5f);
  };
  return (q(rgba[0]) << 24) | (q(rgba[1]) << 16) | (q(rgba[2]) << 8) | q(rgba[3]);
}

}  // namespace

struct BourbonContext::Impl {
  // Metal objects. The device is owned by the CAMetalLayer (SDL created it); we
  // own only the command queue we allocate against it.
  CA::MetalLayer* metal_layer = nullptr;
  MTL::Device* device = nullptr;
  NS::SharedPtr<MTL::CommandQueue> mtl_queue;

  // Bourbon contexts (single window, single frame in flight).
  std::unique_ptr<bourbon::CoreContext> core;
  std::unique_ptr<bourbon::SGContext> sg_context;
  std::unique_ptr<bourbon::RendererContext> renderer_context;
  std::unique_ptr<bourbon::Swapchain> swapchain;
  bourbon::CommandQueue* queue = nullptr;  // owned by core.

  // Two dataflow graphs: the model graph (scene state) and the render graph
  // (pass chain), each with its own evaluator and scratch heap.
  std::unique_ptr<bourbon::TaskGraph> model_graph;
  std::unique_ptr<bourbon::TaskGraphEvaluator> model_graph_evaluator;
  std::unique_ptr<bourbon::PersistentHeap> model_heap;
  std::unique_ptr<bourbon::TaskGraph> render_graph;
  std::unique_ptr<bourbon::TaskGraphEvaluator> render_graph_evaluator;
  std::unique_ptr<bourbon::PersistentHeap> render_heap;

  std::unique_ptr<bourbon::World> world;

  // Camera + projection (live via DGInputs, updated per frame).
  bourbon::RefPtr<bourbon::MatrixProjection> projection;
  bourbon::RefPtr<bourbon::Camera> camera;

  // Draw submission and pipeline tables.
  std::unique_ptr<bourbon::DrawSubmission> draw_submission;
  std::unique_ptr<bourbon::BXDFIntegrationPipelines> bxdf_pipelines;
  std::unique_ptr<bourbon::MaterialEvaluationPipelines> material_pipelines;

  // Forward pass graph: clear -> integrate -> resolve into the drawable.
  std::unique_ptr<DrawableImageSource> drawable_source;
  std::unique_ptr<bourbon::ClearFramebufferTask> clear_task;
  std::unique_ptr<bourbon::ForwardIntegrator> integrator;
  std::unique_ptr<bourbon::ResolveTask> resolve_task;
  bourbon::Timestamp prior_render_signal{};

  // A LoadAction=Load framebuffer used to overlay ImGui on the resolved image.
  std::unique_ptr<bourbon::Framebuffer> ui_framebuffer;

  const mjModel* model = nullptr;
  Eigen::Array4f clear_color = {0.12f, 0.14f, 0.18f, 1.0f};

  // Retained scene: one entry per model geom (size ngeom; entries for
  // unsupported geom types have a null node). Plus a per-colour flat-material
  // cache and a single hard-coded distant light (Stage 1).
  std::vector<Renderable> geoms;
  const bourbon::BXDFClass* pbr_bxdf = nullptr;
  const bourbon::PatternClass* pbr_pattern = nullptr;
  std::map<uint32_t, bourbon::RefPtr<bourbon::Material>> material_cache;
  std::vector<bourbon::SGNode*> light_nodes;
  std::vector<bourbon::RefPtr<bourbon::DistantLight>> lights;

  std::chrono::steady_clock::time_point last_frame;
  bool have_last_frame = false;
  double fps = 0.0;

  void DrainGpu() {
    if (!core || !queue) return;
    auto fence = bourbon::CommandBuffer::Create(*queue, *core);
    fence->enqueue();
    fence->submit();
    fence->metal_command_buffer()->waitUntilCompleted();
  }

  void BuildPassGraph() {
    bourbon::DeviceImage<2>::Extent extent = ToExtent(swapchain->extent());

    clear_task = bourbon::ClearFramebufferTask::Create(
        extent, kColorFormat, /*sample_count=*/1, clear_color, *render_graph,
        *queue, *renderer_context);

    bourbon::Integrator::Params integrator_params = {
        .extent = extent,
        .color_format = kColorFormat,
        .depth_format = kDepthFormat,
        .sample_count = 1,
        .oit = bourbon::OITKind::None,
        .tap = bourbon::Tap::Color,
        .clustered = false,
    };
    bourbon::ForwardVizTask::Params forward_viz = {
        .surface = bourbon::Token::Get("lit"),
        .texture = bourbon::Token::Get("material"),
    };

    integrator = bourbon::ForwardIntegrator::Create(
        clear_task->color(), camera.get(), world.get(),
        draw_submission->draws(), draw_submission->shadowDraws(),
        bxdf_pipelines.get(), forward_viz, integrator_params, *render_heap,
        *render_graph, *queue, *renderer_context);

    drawable_source = std::make_unique<DrawableImageSource>(*render_graph);

    resolve_task = bourbon::ResolveTask::Create(
        extent, swapchain->format(), drawable_source->output(),
        integrator->output(), *camera, *render_graph, *renderer_context);
  }

  // Maps a MuJoCo geom to a bourbon SDF primitive, with sizes baked in from
  // mjModel::geom_size (index 1 is the half-length for capsule/cylinder). Ray-
  // marched SDFs require rigid node transforms, so all sizing lives here, never
  // in the node scale. Returns null for geom types not yet supported (plane,
  // mesh, hfield, ...).
  bourbon::RefPtr<bourbon::Shape> MakeShapeForGeom(const mjModel* m, int i) {
    const mjtNum* size = m->geom_size + 3 * i;
    const float a = static_cast<float>(size[0]);
    const float b = static_cast<float>(size[1]);
    const float c = static_cast<float>(size[2]);
    bourbon::TaskHeap& h = *model_heap;
    bourbon::TaskGraph& g = *model_graph;
    bourbon::SGContext& sg = *sg_context;
    switch (m->geom_type[i]) {
      case mjGEOM_SPHERE:
        return bourbon::SDF3D::Create(
            bourbon::SDF3D::SphereParams{.radius = a}, h, g, sg);
      case mjGEOM_CAPSULE:
        return bourbon::SDF3D::Create(
            bourbon::SDF3D::CapsuleParams{.radius = a, .height = 2.0f * b}, h, g,
            sg);
      case mjGEOM_CYLINDER:
        return bourbon::SDF3D::Create(
            bourbon::SDF3D::CylinderParams{.radius = a, .height = 2.0f * b}, h,
            g, sg);
      case mjGEOM_BOX:
        return bourbon::SDF3D::Create(
            bourbon::SDF3D::CubeParams{
                .width = 2.0f * a, .height = 2.0f * b, .depth = 2.0f * c},
            h, g, sg);
      case mjGEOM_ELLIPSOID:
        return bourbon::SDF3D::Create(
            bourbon::SDF3D::EllipsoidParams{
                .radius_x = a, .radius_y = b, .radius_z = c},
            h, g, sg);
      default:
        return {};  // plane, mesh, hfield, etc.: later stages.
    }
  }

  // Removes all retained scene nodes (geoms + light) from the world model.
  void ClearScene() {
    if (world) {
      world->model().destroyAllRootNodes();
    }
    geoms.clear();
    material_cache.clear();
    pbr_bxdf = nullptr;
    pbr_pattern = nullptr;
    light_nodes.clear();
    lights.clear();
  }

  // Returns a flat glTF-PBR material for `rgba`, creating and caching one per
  // distinct (quantized) colour so geoms that share a colour share a material.
  bourbon::RefPtr<bourbon::Material> GetOrCreateMaterial(const float rgba[4]) {
    const uint32_t key = QuantizeRgba(rgba);
    auto it = material_cache.find(key);
    if (it != material_cache.end()) return it->second;

    Eigen::Array4f base_color = {rgba[0], rgba[1], rgba[2], rgba[3]};
    bourbon::Pattern::Params params = {
        {bourbon::Token::Get("baseColor"), &base_color}};
    auto material = bourbon::Material::Create(
        pbr_bxdf,
        bourbon::Pattern::Create(*pbr_pattern, params, {},
                                 bourbon::SemiTransparencyKind::Opaque,
                                 *model_heap, *model_graph, *sg_context),
        *model_heap, *model_graph, *sg_context);
    material_cache[key] = material;
    return material;
  }

  // Builds the retained scene for `m`: a shared flat material, one node/shape
  // per supported geom (transforms updated per frame), and one distant light.
  void BuildScene(const mjModel* m) {
    model = m;
    if (!m) return;
    bourbon::TaskHeap& h = *model_heap;
    bourbon::TaskGraph& g = *model_graph;
    bourbon::SGContext& sg = *sg_context;
    bourbon::Model& scene = world->model();

    // Flat glTF-PBR material templates; GetOrCreateMaterial() instantiates one
    // per distinct geom colour (Stage 3 replaces these with per-material
    // scalars/textures).
    pbr_bxdf = renderer_context->findBXDF(bourbon::Token::Get("glTFPbrBXDF"));
    pbr_pattern =
        renderer_context->findPattern(bourbon::Token::Get("glTFPbrPattern"));

    geoms.assign(m->ngeom, Renderable{});
    for (int i = 0; i < m->ngeom; ++i) {
      bourbon::RefPtr<bourbon::Shape> shape = MakeShapeForGeom(m, i);
      if (!shape) continue;
      float rgba[4];
      EffectiveGeomRgba(m, i, rgba);
      bourbon::RefPtr<bourbon::Material> material = GetOrCreateMaterial(rgba);
      bourbon::SGNode* node = scene.createNode();
      scene.addRootNode(*node);
      auto xform = bourbon::MatrixTransformer::Create(g);
      node->setTransformer(xform);
      node->addShape(shape, material, h, g, sg);
      geoms[i].node = node;
      geoms[i].xform = xform;
      geoms[i].shape = shape;
      geoms[i].material = material;
    }

    BuildLight();
  }

  // Adds one distant light travelling in world direction `dir` at `lux`
  // illuminance. Direction only (from is the origin; DistantLight is placement-
  // independent), so an identity node transform suffices.
  void AddDistantLight(const Eigen::Vector3f& dir, float lux) {
    bourbon::TaskHeap& h = *model_heap;
    bourbon::TaskGraph& g = *model_graph;
    bourbon::SGContext& sg = *sg_context;
    bourbon::Model& scene = world->model();

    bourbon::DistantLight::Params p;
    p.intensity = {lux, bourbon::LightSource::Intensity::Unit::Lux};
    // Colour defaults to unit-luminance white RGB.
    p.from = bourbon::Point<bourbon::Space::Object>{0.0f, 0.0f, 0.0f, 1.0f};
    p.to = bourbon::Point<bourbon::Space::Object>{dir.x(), dir.y(), dir.z(),
                                                  1.0f};
    auto light = bourbon::DistantLight::Create(p, h, g, sg);

    bourbon::SGNode* node = scene.createNode();
    node->setTransformer(bourbon::MatrixTransformer::Create(g));
    node->addLightSource(light, h, g, sg);
    scene.addRootNode(*node);

    lights.push_back(std::move(light));
    light_nodes.push_back(node);
  }

  // Hard-coded key + fill distant lights so the scene is neither black nor
  // starkly single-sided (Stage 2 replaces these with the model's own lights
  // and image-based lighting). MuJoCo is z-up; the key rakes down across a
  // standing model and the dimmer fill lifts the opposite side.
  void BuildLight() {
    // Key: clear-noon-sun illuminance, paired with the camera's sunny-16
    // photometric exposure (see the Camera setup) so the scene lands in range.
    AddDistantLight(Eigen::Vector3f(0.3f, 0.5f, -1.0f), 100000.0f);
    // Fill from the opposite side and less steep, at ~1/5 the key, to soften
    // the shadow side without flattening the form.
    AddDistantLight(Eigen::Vector3f(-0.5f, -0.4f, -0.6f), 20000.0f);
  }
};

BourbonContext::BourbonContext(void* metal_layer)
    : impl_(std::make_unique<Impl>()) {
  if (!metal_layer) {
    mju_error("BourbonContext: null Metal layer");
  }
  Impl& s = *impl_;
  s.metal_layer = reinterpret_cast<CA::MetalLayer*>(metal_layer);
  s.device = s.metal_layer->device();
  if (!s.device) {
    mju_error("BourbonContext: CAMetalLayer has no MTLDevice");
  }

  // The Metal toolchain must be reachable before any pipeline is built.
  llair::setPathToTools(MetalToolsPath());

  s.mtl_queue = NS::TransferPtr(s.device->newCommandQueue());
  s.core = std::make_unique<bourbon::CoreContext>(s.device, s.mtl_queue.get());
  s.sg_context = std::make_unique<bourbon::SGContext>(*s.core);
  s.renderer_context =
      std::make_unique<bourbon::RendererContext>(*s.core, *s.sg_context);
  s.swapchain = bourbon::Swapchain::Create(s.metal_layer, *s.core);
  s.queue = s.core->getCommandQueue();

  // Model + render graphs, evaluators, and heaps. TaskGraph::Create does NOT
  // synthesize a fallback allocator when passed nullptr, so an explicit
  // memory_resource is required; new_delete_resource is an unbounded static
  // singleton (no lifetime management needed).
  s.model_graph =
      bourbon::TaskGraph::Create(*s.core, std::pmr::new_delete_resource());
  s.model_graph_evaluator = bourbon::TaskGraphEvaluator::Create(*s.model_graph);
  s.model_heap = std::make_unique<bourbon::PersistentHeap>(*s.core);
  s.render_graph =
      bourbon::TaskGraph::Create(*s.core, std::pmr::new_delete_resource());
  s.render_graph_evaluator =
      bourbon::TaskGraphEvaluator::Create(*s.render_graph);
  s.render_heap = std::make_unique<bourbon::PersistentHeap>(*s.core);

  s.world = bourbon::World::Create(*s.model_graph, *s.render_graph,
                                   *s.renderer_context);

  // Residency sets are registered once for the lifetime of the queue.
  s.queue->addResidencySet(s.model_heap->residencySet());
  s.queue->addResidencySet(s.render_heap->residencySet());
  s.queue->addResidencySet(s.world->residencySet());

  // Camera + projection. Both are live via DGInputs and updated each frame; the
  // initial values are placeholders.
  bourbon::DeviceImage<2>::Extent extent = ToExtent(s.swapchain->extent());
  s.projection = bourbon::MatrixProjection::Create(
      bourbon::MatrixProjection::FrustumParams{}, *s.model_heap, *s.model_graph,
      *s.sg_context);
  // Photometric exposure with the "sunny-16" preset (f/16, 1/125 s, ISO 100),
  // matching bourbon's own viewer defaults. This is calibrated to correctly
  // expose a scene lit by a ~100k-lux (clear-noon-sun) distant light — see
  // BuildLight(). The two must be tuned together: a physically bright light
  // with these camera settings lands in range, whereas an arbitrary linear
  // exposure would not.
  s.camera = bourbon::Camera::Create(
      bourbon::Camera::Params{
          .extent = extent,
          .projection = s.projection,
          .exposure_mode = bourbon::Camera::ExposureMode::Photometric,
          .f_number = 16.0f,
          .shutter_time = 1.0f / 125.0f,
          .iso = 100.0f,
          .ev_compensation = 0.0f,
      },
      *s.model_heap, *s.model_graph, *s.sg_context);
  s.world->model().setViewCamera(s.camera.get());

  s.draw_submission = bourbon::DrawSubmission::Create(
      *s.world, *s.camera, *s.render_graph, *s.renderer_context);
  s.draw_submission->setMode(bourbon::DrawSubmission::Mode::Direct);
  s.bxdf_pipelines = bourbon::BXDFIntegrationPipelines::Create(
      *s.world, *s.render_graph, *s.renderer_context);
  s.material_pipelines = bourbon::MaterialEvaluationPipelines::Create(
      *s.world, *s.render_graph, *s.renderer_context);

  s.BuildPassGraph();

  // A framebuffer for the ImGui overlay: LoadAction=Load preserves the resolved
  // scene already written into the drawable.
  auto* ui_attachment = bourbon::RenderPassAttachment::Get(
      s.swapchain->format(), MTL::LoadActionLoad, MTL::StoreActionStore,
      *s.core);
  auto* ui_render_pass = bourbon::RenderPass::Get(
      bourbon::RenderPassColorAttachments::AttachmentsVector{ui_attachment},
      *s.core);
  s.ui_framebuffer = bourbon::Framebuffer::Create(ui_render_pass, *s.core);

  ImGui_ImplMetalCPP_Init(s.device);
}

BourbonContext::~BourbonContext() {
  impl_->DrainGpu();
  ImGui_ImplMetalCPP_Shutdown();
}

void BourbonContext::Init(const mjModel* model) {
  impl_->DrainGpu();
  impl_->ClearScene();
  impl_->BuildScene(model);
}

void BourbonContext::SetClearColor(float r, float g, float b, float a) {
  impl_->clear_color = Eigen::Array4f{r, g, b, a};
}

void BourbonContext::RenderFrame(const mjModel* model, mjData* data,
                                 mjvCamera* mj_camera, int width, int height) {
  Impl& s = *impl_;

  // Must be first each frame.
  s.core->advanceCompletionTimestamp();

  std::unique_ptr<bourbon::Drawable> drawable = s.swapchain->getNextDrawable();
  if (!drawable) {
    ImGui::EndFrame();
    return;
  }
  bourbon::DeviceImage<2>::Extent extent = ToExtent(s.swapchain->extent());

  s.ui_framebuffer->setColorAttachment(0, drawable->image());
  ImGui_ImplMetalCPP_NewFrame(s.ui_framebuffer->metal_render_pass_descriptor());
  ImGui::Render();

  auto frame_commands = bourbon::CommandBuffer::Create(*s.queue, *s.core);

  // --- Update the camera from the MuJoCo camera. ---
  if (model != nullptr && data != nullptr && mj_camera != nullptr) {
    mjvGLCamera gl = mjv_camera2GLCamera(model, data, mj_camera);
    Eigen::Vector3f eye(gl.pos[0], gl.pos[1], gl.pos[2]);
    Eigen::Vector3f fwd(gl.forward[0], gl.forward[1], gl.forward[2]);
    Eigen::Vector3f up(gl.up[0], gl.up[1], gl.up[2]);
    Eigen::Matrix4f world_to_eye = bourbon::LookInDirection(eye, fwd, up);
    Eigen::Affine3f a;
    a.matrix() = world_to_eye;
    Eigen::Affine3f a_inv;
    a_inv.matrix() = world_to_eye.inverse();
    s.camera->transform().setValueIfChanged(a);
    s.camera->inverse_transform().setValueIfChanged(a_inv);
    s.camera->extent().setValueIfChanged(extent);

    const float aspect = height > 0 ? static_cast<float>(width) / height : 1.0f;
    const float bottom = gl.frustum_bottom;
    const float top = gl.frustum_top;
    const float near = gl.frustum_near;
    const float far = gl.frustum_far;
    const float halfwidth =
        gl.frustum_width ? gl.frustum_width : 0.5f * aspect * (top - bottom);
    const float left = gl.frustum_center - halfwidth;
    const float right = gl.frustum_center + halfwidth;
    Eigen::Matrix4f p = gl.orthographic
                            ? bourbon::Ortho(left, right, bottom, top, near, far)
                                  .matrix()
                            : bourbon::Frustum(left, right, bottom, top, near,
                                               far)
                                  .matrix();
    s.projection->eye_to_clip().setValueIfChanged(
        bourbon::Transformation<bourbon::Space::Eye, bourbon::Space::Clip>{
            p, p.inverse()});
    s.integrator->setDepthRange(near, far);
  }

  // --- Push per-geom world transforms into the retained scene. ---
  if (model != nullptr && data != nullptr) {
    const int n = static_cast<int>(s.geoms.size());
    for (int i = 0; i < n; ++i) {
      if (!s.geoms[i].node) continue;
      s.geoms[i].xform->local_matrix().setValueIfChanged(
          GeomAffine(data->geom_xpos + 3 * i, data->geom_xmat + 9 * i));
    }
  }

  // --- Evaluate the model (scene state) graph. ---
  s.model_heap->free();
  s.model_heap->allocate();
  s.model_graph_evaluator->evaluate(*s.model_heap, *s.core);

  // --- Feed the drawable into the render graph and evaluate it. ---
  {
    bourbon::Promise<bourbon::DeviceImage<2>> promise;
    promise.set_value(drawable->image());
    s.drawable_source->set(promise.get_future().share());
  }
  s.clear_task->extent_in().setValue(extent);
  s.integrator->setExtent(extent);

  s.render_heap->free();
  s.render_heap->allocate();
  s.prior_render_signal = s.render_graph_evaluator->evaluate({
      .task_heap = *s.render_heap,
      .context = *s.core,
      .prior_signal = s.prior_render_signal,
  });

  // --- Overlay ImGui onto the resolved drawable, then present. ---
  auto encoder =
      bourbon::RenderCommandEncoder::Create(*frame_commands, *s.ui_framebuffer);
  ImGui_ImplMetalCPP_RenderDrawData(ImGui::GetDrawData(),
                                    frame_commands->metal_command_buffer(),
                                    encoder->metal_render_command_encoder());
  encoder->end();

  frame_commands->signalCompletion();
  frame_commands->present(std::shared_ptr<bourbon::Drawable>(std::move(drawable)));
  frame_commands->submit();

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
