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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <map>
#include <memory>
#include <memory_resource>
#include <string>
#include <system_error>
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
#include <BourbonSG/LightSources/EnvironmentLight.h>
#include <BourbonSG/LightSources/PointLight.h>
#include <BourbonSG/LightSources/SpotLight.h>
#include <BourbonSG/Material.h>
#include <BourbonSG/MatrixTransformer.h>
#include <BourbonSG/Model.h>
#include <BourbonSG/Pattern.h>
#include <BourbonSG/Projections/MatrixProjection.h>
#include <BourbonSG/SGContext.h>
#include <BourbonSG/SGNode.h>
#include <BourbonSG/SGObjectKind.h>
#include <BourbonSG/ShadowFilter.h>
#include <BourbonSG/ShadowSource.h>
#include <BourbonSG/ShadowSourceInstance.h>
#include <BourbonSG/ShadowSources/CubeShadow.h>
#include <BourbonSG/ShadowSources/DirectionalShadow.h>
#include <BourbonSG/ShapeInstance.h>
#include <BourbonSG/ShapeVar.h>
#include <BourbonSG/ShapeVarKind.h>
#include <BourbonSG/Shapes/SDF3D.h>
#include <BourbonSG/Shapes/TriangleMesh.h>
#include <BourbonSG/TextureMapSet.h>
#include <BourbonSG/TextureMaps/CubeTextureMap.h>
#include <BourbonSG/TextureMaps/TextureCoordMaps.h>
#include <BourbonSG/TextureMaps/TextureMap2D.h>

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

// Returns the directory holding the Metal toolchain, or an empty string.
// Asks xcrun rather than searching PATH: the toolchain is not on the default
// PATH. A Finder-launched .app inherits /usr/bin so xcrun itself resolves, and
// DEVELOPER_DIR is unset there, so xcrun falls back to the xcode-select
// default -- which is the right answer.
std::string MetalToolsPathFromXcrun() {
  FILE* pipe = popen("xcrun -f metal 2>/dev/null", "r");
  if (!pipe) {
    return "";
  }
  std::string out;
  char buf[1024];
  while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
    out += buf;
  }
  if (pclose(pipe) != 0 || out.empty()) {
    return "";
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  if (out.empty()) {
    return "";
  }
  return std::filesystem::path(out).parent_path().string();
}

// Directory containing the Metal toolchain, resolved once on first use.
//
// The CMake-baked path is tried before xcrun to keep a subprocess off the
// common path, but only if it still points at a real toolchain: on recent
// macOS the toolchain lives on a cryptex mount whose path carries a
// per-mount suffix, so the configure-time value goes stale across a reboot or
// a Metal toolchain update -- and is meaningless inside a relocated .app.
// The existence check is what makes that case self-heal via xcrun instead of
// hard-failing.
const char* MetalToolsPath() {
  static const std::string* const resolved = [] {
    std::error_code ec;
    // An explicit override that is wrong is reported rather than silently
    // ignored: falling through to xcrun would hide the user's mistake behind a
    // toolchain that happens to work.
    if (const char* env = std::getenv("MUJOCO_BOURBON_LLAIR_TOOLS")) {
      if (!std::filesystem::exists(std::filesystem::path(env) / "metal", ec)) {
        mju_error(
            "Bourbon renderer: MUJOCO_BOURBON_LLAIR_TOOLS is set to '%s', but "
            "there is no 'metal' compiler in that directory. Point it at the "
            "directory containing 'metal' (see 'xcrun -f metal'), or unset it "
            "to auto-detect.",
            env);
      }
      return new std::string(env);
    }
    if (std::filesystem::exists(
            std::filesystem::path(MUJOCO_BOURBON_LLAIR_TOOLS_PATH) / "metal",
            ec)) {
      return new std::string(MUJOCO_BOURBON_LLAIR_TOOLS_PATH);
    }
    std::string from_xcrun = MetalToolsPathFromXcrun();
    if (!from_xcrun.empty()) {
      return new std::string(std::move(from_xcrun));
    }
    mju_error(
        "Bourbon renderer: cannot find the Metal toolchain (the 'metal' "
        "compiler), which it needs to build shader pipelines at runtime. "
        "Neither the build-time path '%s' nor 'xcrun -f metal' resolved. "
        "Install it with 'xcodebuild -downloadComponent MetalToolchain' (the "
        "Command Line Tools alone are not sufficient), or point "
        "MUJOCO_BOURBON_LLAIR_TOOLS at the directory containing 'metal'.",
        MUJOCO_BOURBON_LLAIR_TOOLS_PATH);
    return new std::string();  // Unreachable: mju_error does not return.
  }();
  return resolved->c_str();
}

bourbon::DeviceImage<2>::Extent ToExtent(CGSize size) {
  return bourbon::DeviceImage<2>::Extent{static_cast<unsigned>(size.width),
                                         static_cast<unsigned>(size.height)};
}

// A retained light: an owned SGNode whose transform places/orients the light
// each frame (bourbon lights read position/direction from the node transform,
// not from world-space params). `kind` selects the concrete downcast for the
// per-frame colour/intensity update. `intensity_value`/`intensity_unit` are the
// magnitude in the unit bourbon requires for this kind, computed once at build.
struct Light {
  bourbon::SGNode* node = nullptr;  // owned by world->model()
  bourbon::RefPtr<bourbon::MatrixTransformer> xform;
  bourbon::RefPtr<bourbon::LightSource> source;
  bourbon::SGObjectKind kind = bourbon::SGObjectKind::LightSourceMax;
  float intensity_value = 0.0f;
  bourbon::LightSource::Intensity::Unit intensity_unit =
      bourbon::LightSource::Intensity::Unit::Lux;
  bool is_headlight = false;
  // An image-based (environment) light. Its placement frame and its radiance
  // are both baked at build time, so it is skipped by the per-frame update and
  // is not index-aligned with mjModel's lights (it is appended last).
  bool is_environment = false;

  // Shadow state. A shadow source is attached to this same node (so it inherits
  // the light's placement frame) on demand and removed when shadows are turned
  // off; `casts_shadow` gates that from light_castshadow. Projection params are
  // baked at construction of the ShadowSource (they are NOT DGInputs), so they
  // are computed once at build and cached here.
  bool casts_shadow = false;
  bourbon::RefPtr<bourbon::ShadowSource> shadow;
  bourbon::ShadowSourceInstance* shadow_instance = nullptr;
  unsigned shadow_size = 1024;
  float shadow_near = 0.01f;
  float shadow_far = 100.0f;
  float shadow_coverage = 1.0f;  // directional ortho half-extent (width=height=2x)
  float shadow_coneangle = 180.0f;  // spot shadow full cone angle (degrees)
  // Depth-comparison bias, in bourbon's normalized-distance units (a fraction of
  // far-near). Computed from a fixed world-space offset and the fitted frustum
  // span so it stays physical regardless of scene size; see BuildLights.
  Eigen::Array2f shadow_bias = {0.005f, 0.05f};
};

// Classic-scene fallback illumination (used when a model authors no physical
// light_intensity, i.e. brightness lives in the [0,1] light_diffuse colour, as
// in humanoid.xml/car.xml). These are calibrated against the camera's
// "sunny-16" photometric exposure (see the Camera setup) so a classic scene
// lands in range; they are the Stage-2 successors of the old hard-coded
// key/fill lights. Directional lights get a fixed illuminance; punctual lights
// get a candela derived from their build-time distance to the scene centre so
// the illuminance they deliver near the model matches the directional target.
constexpr float kClassicDirectionalLux = 100000.0f;
// Punctual (spot/point) classic-scene intensity, in candela. With MuJoCo's
// default attenuation {1,0,0} bourbon applies NO distance falloff (the shader
// factor is clamp(1/att0, 0, 1) = 1), so this candela value is used directly as
// the on-axis illuminance -- it is NOT divided by distance^2. Kept a bit under
// the directional target so two overlapping spots plus the headlight stay in
// range under the sunny-16 exposure.
constexpr float kClassicPunctualCandela = 25000.0f;

constexpr float kPi = 3.14159265358979323846f;

// --- Image-based (environment) lighting -------------------------------------
//
// Bourbon has no visible-background pass, so the environment light is purely
// the indirect/ambient term: it is what stops the unlit sides of a model from
// going black under a PBR integrator, and it is the closest analogue MuJoCo's
// ambient terms have here.
//
// The source is always a lat-long (equirectangular) image, even for a MuJoCo
// skybox cube: resampling the cube on the host is a few lines and keeps a
// single code path whose orientation we control exactly, whereas the
// CubeEnvironmentLight shader path negates its sample direction and bypasses
// the BXDF's own indirect evaluation.
//
// Sizes are powers of two: LatLongEnvironmentLight derives its radiance mip
// count as log2(height). The source is deliberately small — it is only ever
// read to bake the irradiance/radiance maps, never sampled for display.
constexpr unsigned kEnvSourceHeight = 128;      // source is 2H x H
constexpr unsigned kEnvIrradianceHeight = 64;
constexpr unsigned kEnvRadianceHeight = 128;
constexpr std::array<unsigned, 2> kEnvIrradianceSampleCount = {128, 32};

// Ambient irradiance a fully white (1,1,1) skybox delivers, as a fraction of
// the scene's reference illuminance (see ReferenceIlluminance). A skybox image
// carries no photometric units, so this is the calibration that puts it in
// range under the camera's sunny-16 exposure. When there is no skybox the
// fraction is MuJoCo's own ambient instead, which needs no fudge factor: the
// classic renderer adds `ambient * albedo` next to `diffuse * NdotL * albedo`,
// so an environment delivering `ambient * reference` lux reproduces exactly
// that ratio.
constexpr float kSkyboxIrradianceFraction = 0.15f;

// Radiance of the lower hemisphere relative to the upper one in the
// synthesized gradient. Below 1 so the fill keeps an up/down cue rather than
// flattening the model the way a uniform ambient does.
constexpr float kEnvGroundFraction = 0.3f;

// Neutral ambient fill (as an effective [0,1] grey ambient coefficient) used
// when a model authors no ambient and carries no skybox. Without it, a point
// occluded from every punctual light receives zero radiance and reads as pure
// black -- the shadows looked correct in shape but crushed. The Filament
// backend never has this problem because it always attaches a fallback
// environment light (~5000 lux; model_lights.cc). 0.05 * kClassicDirectionalLux
// = 5000 lux matches that, so classic skybox-less scenes get the same soft fill.
constexpr float kClassicAmbientFraction = 0.05f;

// PCSS penumbra size (in shadow-map texels) for soft shadow edges. bourbon's
// PCSS estimates a blocker ratio and widens the PCF kernel from this (search
// clamped to 4, kernel to 6 -> up to 13x13 texels), so 4 is the largest value
// that still changes the result; without it the fitted, high-resolution shadow
// map produces razor-hard, aliased edges.
constexpr float kShadowPenumbraSize = 4.0f;

// How much of a light a shadow removes (bourbon ShadowSource intensity). At 1.0
// a shadowed surface keeps only ambient/environment fill and reads near-black
// against directly lit neighbours; letting a fraction of the direct light
// through lifts the shadow to a plausible grey without washing out the scene.
constexpr float kShadowIntensity = 0.8f;

// Upper bound on the shadow-map edge length. Larger maps give finer edge steps
// (less pixelation), at 2 bytes/texel: 8192^2 Depth16 is 128 MB per shadow.
constexpr unsigned kMaxShadowSize = 8192;

// sRGB electro-optical transfer function (decode to linear).
float SrgbToLinear(float c) {
  return c <= 0.04045f ? c / 12.92f
                       : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

int ClampInt(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Reads texel `texel` (a flat index into texture `texid`'s data) as linear RGB
// in [0,1], decoding sRGB when the texture declares it. Single-channel data is
// broadcast to grey; the alpha channel of RGBA data is ignored (an environment
// has no transparency).
Eigen::Array3f TexelLinear(const mjModel* m, int texid, size_t texel) {
  const int nchannel = m->tex_nchannel[texid];
  const mjtByte* p = m->tex_data + m->tex_adr[texid] + texel * nchannel;
  const bool srgb = m->tex_colorspace[texid] == mjCOLORSPACE_SRGB;
  Eigen::Array3f c;
  for (int k = 0; k < 3; ++k) {
    const float v = (nchannel > k ? p[k] : p[0]) / 255.0f;
    c[k] = srgb ? SrgbToLinear(v) : v;
  }
  return c;
}

// Point-samples a MuJoCo cube/skybox texture along `d`, in the OpenGL/Metal
// cube-map convention (identical in both). Faces are w*w and stacked in GL
// face order (+X,-X,+Y,-Y,+Z,-Z); a square texture (height == width) is a
// single face repeated on all six (render_context.c:1525).
//
// A MuJoCo skybox cube is authored Y-UP, not z-up: the classic renderer binds
// it with object-linear texgen s=x, t=z, r=-y -- "rotate 90 deg around X"
// (render_gl3.c:233-241) -- and the builtin cube generator agrees, filling
// face 2 (+Y) with "up" and face 3 (-Y) with "down" (user_objects.cc:4925).
// That rotation is exactly the environment light's own frame, so `d` here is
// the direction in the light's OBJECT space, not in world space.
Eigen::Array3f SampleCubeTexture(const mjModel* m, int texid,
                                 const Eigen::Vector3f& d) {
  const int w = m->tex_width[texid];
  const int h = m->tex_height[texid];
  const float ax = std::fabs(d.x()), ay = std::fabs(d.y()), az = std::fabs(d.z());
  int face;
  float sc, tc, ma;
  if (ax >= ay && ax >= az) {
    ma = ax;
    if (d.x() > 0) { face = 0; sc = -d.z(); } else { face = 1; sc = d.z(); }
    tc = -d.y();
  } else if (ay >= az) {
    ma = ay;
    if (d.y() > 0) { face = 2; tc = d.z(); } else { face = 3; tc = -d.z(); }
    sc = d.x();
  } else {
    ma = az;
    if (d.z() > 0) { face = 4; sc = d.x(); } else { face = 5; sc = -d.x(); }
    tc = -d.y();
  }
  if (ma < 1e-9f) ma = 1e-9f;
  const int col = ClampInt(static_cast<int>(0.5f * (sc / ma + 1.0f) * w), 0, w - 1);
  const int row = ClampInt(static_cast<int>(0.5f * (tc / ma + 1.0f) * w), 0, w - 1);
  const size_t face_offset =
      (h == w) ? 0 : static_cast<size_t>(face) * w * w;
  return TexelLinear(m, texid, face_offset + static_cast<size_t>(row) * w + col);
}

// Point-samples a MuJoCo 2D texture as an equirectangular environment along
// world direction `d`: longitude runs counter-clockwise from +x about the
// world z axis, and row 0 is the zenith. MuJoCo does not pin a convention for
// mjLIGHT_IMAGE textures, so this is ours.
Eigen::Array3f SampleLatLongTexture(const mjModel* m, int texid,
                                    const Eigen::Vector3f& d) {
  const int w = m->tex_width[texid];
  const int h = m->tex_height[texid];
  const float u = std::atan2(d.y(), d.x()) / (2.0f * kPi) + 0.5f;
  const float z = d.z() < -1.0f ? -1.0f : (d.z() > 1.0f ? 1.0f : d.z());
  const float v = std::acos(z) / kPi;
  const int col = ClampInt(static_cast<int>(u * w), 0, w - 1);
  const int row = ClampInt(static_cast<int>(v * h), 0, h - 1);
  return TexelLinear(m, texid, static_cast<size_t>(row) * w + col);
}

// Inverse of MaterialX's mx_latlong_projection (mx_microfacet_specular.glsl),
// which is the mapping the baked environment maps are ultimately sampled with:
// v = -asin(y)/pi + 0.5, u = atan2(x, -z)/2pi + 0.5. Returns the direction in
// the environment light's OBJECT space, which is y-up.
Eigen::Vector3f LatLongDirection(float u, float v) {
  const float y = std::sin(kPi * (0.5f - v));
  const float r2 = 1.0f - y * y;
  const float r = std::sqrt(r2 > 0.0f ? r2 : 0.0f);
  const float theta = 2.0f * kPi * (u - 0.5f);
  return Eigen::Vector3f(r * std::sin(theta), y, -r * std::cos(theta));
}

// Object->world transform for the environment light's node. The shader samples
// the environment with the direction taken into the node's object space, and
// MaterialX's lat-long projection is y-up, so this maps object +y onto MuJoCo's
// world +z: a rotation of +90 degrees about x (y->z, z->-y).
Eigen::Affine3f EnvironmentFrame() {
  return Eigen::Affine3f(
      Eigen::AngleAxisf(0.5f * kPi, Eigen::Vector3f::UnitX()));
}

// The illuminance the scene's brightest light delivers, used to put the
// environment's radiance on the same photometric scale as the punctual lights
// (whose classic-scene magnitudes are themselves calibrated against the
// camera's sunny-16 exposure).
float ReferenceIlluminance(const mjModel* m) {
  float best = 0.0f;
  for (int i = 0; i < m->nlight; ++i) {
    if (m->light_type[i] == mjLIGHT_DIRECTIONAL) {
      best = std::max(best, m->light_intensity[i]);
    }
  }
  return best > 0.0f ? best : kClassicDirectionalLux;
}

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

// Builds a rigid object->world transform from a float position (3) and a
// row-major 3x3 orientation, as carried by mjvGeom (pos/mat are float, and mat
// is row-major just like mjData::geom_xmat). Used for decorations.
Eigen::Affine3f GeomAffineF(const float* pos, const float* mat) {
  Eigen::Matrix3f r;
  r << mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8];
  Eigen::Affine3f a = Eigen::Affine3f::Identity();
  a.linear() = r;
  a.translation() = Eigen::Vector3f(pos[0], pos[1], pos[2]);
  return a;
}

// Any unit vector orthogonal to `v` (a port of mjr_orthoVec): cross with an
// axis that is not parallel to `v`. Used to complete a light's orientation
// frame, whose roll about its own axis is irrelevant for shading.
Eigen::Vector3f OrthoVec(const Eigen::Vector3f& v) {
  Eigen::Vector3f r = v.cross(Eigen::Vector3f(-1.0f, 0.0f, 0.0f));
  if (r.squaredNorm() > 0.01f) return r.normalized();
  return v.cross(Eigen::Vector3f(0.0f, 1.0f, 0.0f)).normalized();
}

// Builds a light node's object->world transform placing it at `pos` with its
// local axis aligned to propagation direction `dir`. Bourbon lights store their
// direction/position in object space and the shader applies the owning node's
// object->world transform, so with default from/to (from=origin, to=+z, giving
// object direction from-to = -z) this frame maps the light's propagation
// direction to `dir` and its position to `pos`: LookInDirection sets row2 = -f,
// so object -z -> +dir and object origin -> pos (see LookInDirection). This is
// the same construction the prior renderer used and it also gives the correct
// DistantLight sign (toward-light = to-from = -dir).
Eigen::Affine3f LightFrame(const Eigen::Vector3f& pos, Eigen::Vector3f dir) {
  if (dir.squaredNorm() < 1e-12f) dir = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
  dir.normalize();
  Eigen::Affine3f a;
  a.matrix() = bourbon::LookInDirection(pos, dir, OrthoVec(dir)).inverse();
  return a;
}

// Reads a MuJoCo float3 (light_xpos/light_xdir row) into an Eigen vector.
Eigen::Vector3f ReadVec3(const mjtNum* p) {
  return Eigen::Vector3f(static_cast<float>(p[0]), static_cast<float>(p[1]),
                         static_cast<float>(p[2]));
}

// Resolves the effective RGBA of an object with material `matid` and per-object
// override `obj_rgba`, mirroring MuJoCo's setMaterial() (engine_vis_visualize.c:
// 225): the material colour is used when the object has a material, but the
// per-object rgba overrides it whenever it differs from the default
// (0.5,0.5,0.5,1) or there is no material. Shared by geoms, flexes and skins.
void EffectiveRgba(const mjModel* m, int matid, const float* obj_rgba,
                   float out[4]) {
  if (matid >= 0) {
    const float* mr = m->mat_rgba + 4 * matid;
    out[0] = mr[0]; out[1] = mr[1]; out[2] = mr[2]; out[3] = mr[3];
  }
  if (obj_rgba[0] != 0.5f || obj_rgba[1] != 0.5f || obj_rgba[2] != 0.5f ||
      obj_rgba[3] != 1.0f || matid < 0) {
    out[0] = obj_rgba[0]; out[1] = obj_rgba[1];
    out[2] = obj_rgba[2]; out[3] = obj_rgba[3];
  }
}

// Effective RGBA of geom `i`.
void EffectiveGeomRgba(const mjModel* m, int i, float out[4]) {
  EffectiveRgba(m, m->geom_matid[i], m->geom_rgba + 4 * i, out);
}

// --- Material scalars --------------------------------------------------------
//
// glTFPbrPattern's uniform defaults are all ZERO: bourbon registers it with
// PatternClass::ParamSpecs whose `value` is null throughout
// (RendererContext.cpp), and PatternClass value-initializes the default
// argument buffer. The MaterialX defaults the gltf_pbr node declares
// (roughness 1, specular 1, ior 1.5, alpha 1) never reach the shader, because
// the pattern drives every one of those BXDF inputs. So every field below must
// be written explicitly -- leaving one out means roughness 0 (mirror-smooth)
// and ior 0, not a sane default.
//
// The pattern has no ambient-occlusion input at all -- neither a scalar nor a
// map slot (its TextureMaps are baseColor/metallicRoughness/transmission/
// emissive/specular/sheen/normal, and `occlusion` is the one gltf_pbr BXDF
// input bourbon's BXDFClass mapping leaves undriven, so it stays at the
// MaterialX default of 1). mjTEXROLE_OCCLUSION therefore has no destination.

// MuJoCo's default geom/material specular is 0.5 whereas glTF's default
// specular weight is 1.0; scaling by two maps one default onto the other, so an
// unconfigured MuJoCo model gets glTF's standard 4% dielectric F0.
constexpr float kSpecularToGltf = 2.0f;

// Index of refraction. MuJoCo has no equivalent field; 1.5 is glTF's default
// and the value the Blinn-Phong specular mapping above is calibrated against.
constexpr float kDefaultIor = 1.5f;

// Upper bound of the OpenGL Blinn-Phong specular exponent, which is what
// mat_shininess (in [0,1]) scales; used only for the roughness fallback.
constexpr float kMaxShininessExponent = 128.0f;

// MuJoCo material defaults for a geom with no material (mjv_initGeom,
// engine_vis_visualize.c:391-394).
constexpr float kDefaultEmission = 0.0f;
constexpr float kDefaultSpecular = 0.5f;
constexpr float kDefaultShininess = 0.5f;

float Clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

// --- Textures ----------------------------------------------------------------

// How a geom's texture coordinates are produced.
//
// bourbon's SDF shapes emit no `st` -- SDF3D.metal never writes the surface
// Geometry's st field, so MaterialGlobals::uv is zero on every ray-marched
// primitive. A texture on a MuJoCo primitive therefore has to be projected from
// the object-space hit position, which is what bourbon's Planar/Spherical/
// Cylindrical coord maps do. Meshes with texcoords, and the plane and height
// field geometry we generate ourselves, carry real UVs and only need the
// texrepeat scale, which TransformUVCoordMap applies (to the derivatives too,
// so mip selection stays correct).
struct TexGen {
  enum class Kind { Uv, Planar, Spherical, Cylindrical };
  Kind kind = Kind::Uv;
  // Uv: st -> scale * st + offset.
  Eigen::Array2f scale = {1.0f, 1.0f};
  Eigen::Array2f offset = {0.0f, 0.0f};
  // Planar: the object-space xy extent one texture tile spans.
  Eigen::Array2f extent = {1.0f, 1.0f};
  float radius = 1.0f;  // Spherical: object z at the pole.
  float height = 1.0f;  // Cylindrical: object z span of one tile.
  // Cube textures ignore everything above and sample with a scaled object-space
  // position, matching the classic renderer's object-linear cube texgen
  // (render_gl3.c:225-230).
  Eigen::Array3f cube_scale = {1.0f, 1.0f, 1.0f};

  std::array<int32_t, 12> Key() const {
    auto q = [](float v) {
      return static_cast<int32_t>(std::lround(v * 4096.0f));
    };
    return {static_cast<int32_t>(kind),
            q(scale[0]),      q(scale[1]),      q(offset[0]),
            q(offset[1]),     q(extent[0]),     q(extent[1]),
            q(radius),        q(height),        q(cube_scale[0]),
            q(cube_scale[1]), q(cube_scale[2])};
  }
};

// Derives the coordinate generation for geom `i` from its type, its size and
// the material's mat_texrepeat/mat_texuniform.
//
// LIMITATION: the spherical and cylindrical coord maps have no post-transform
// on their output st, and bourbon does not chain coord maps, so mat_texrepeat
// cannot be applied to a projected sphere, ellipsoid, capsule or cylinder. (A
// scale+offset on TextureCoordMap2D's output would remove this; worth flagging
// upstream.) Every other path honours it.
TexGen TexGenForGeom(const mjModel* m, int i) {
  TexGen tg;
  const int matid = m->geom_matid[i];
  const float rep0 = matid >= 0 ? m->mat_texrepeat[2 * matid + 0] : 1.0f;
  const float rep1 = matid >= 0 ? m->mat_texrepeat[2 * matid + 1] : 1.0f;
  const bool uniform = matid >= 0 && m->mat_texuniform[matid] != 0;
  const mjtNum* size = m->geom_size + 3 * i;
  const float sx = static_cast<float>(size[0]);
  const float sy = static_cast<float>(size[1]);
  const float sz = static_cast<float>(size[2]);

  tg.cube_scale = uniform ? Eigen::Array3f(sx, sy, sz)
                          : Eigen::Array3f(1.0f, 1.0f, 1.0f);

  // Mirrors the classic renderer's explicit-UV branch (render_gl3.c:130-136): a
  // non-positive repeat means 1, and texuniform re-expresses the repeat in
  // spatial units by multiplying in the geom's own half-extents.
  Eigen::Array2f uv_scale(rep0 > 0.0f ? rep0 : 1.0f, rep1 > 0.0f ? rep1 : 1.0f);
  if (uniform) {
    if (sx > 0.0f) uv_scale[0] *= sx;
    if (sy > 0.0f) uv_scale[1] *= sy;
  }
  // One tile per `rep` over the geom's full extent, or per unit length when
  // texuniform asks for a size-independent scale.
  auto tile = [&](float rep, float half) {
    const float r = rep > 0.0f ? rep : 1.0f;
    if (uniform) return 1.0f / r;
    return half > 0.0f ? 2.0f * half / r : 1.0f / r;
  };

  switch (m->geom_type[i]) {
    case mjGEOM_PLANE:
      tg.kind = TexGen::Kind::Uv;
      tg.scale = uv_scale;
      // BuildPlaneGeometry gives an infinite dimension world-unit UVs rather
      // than 0..1, so the tile size is absolute; the classic renderer then
      // recentres by half a tile (render_gl3.c:145-147).
      if (size[0] <= 0 || size[1] <= 0) tg.offset = {-0.5f, -0.5f};
      break;
    case mjGEOM_HFIELD:
      tg.kind = TexGen::Kind::Uv;
      tg.scale = uv_scale;
      break;
    case mjGEOM_MESH: {
      const int meshid = m->geom_dataid[i];
      if (meshid >= 0 && m->mesh_texcoordadr[meshid] >= 0) {
        tg.kind = TexGen::Kind::Uv;
        tg.scale = uv_scale;
      } else {
        tg.kind = TexGen::Kind::Planar;
        tg.extent = {tile(rep0, sx), tile(rep1, sy)};
      }
    } break;
    case mjGEOM_SPHERE:
      tg.kind = TexGen::Kind::Spherical;
      tg.radius = sx > 0.0f ? sx : 1.0f;
      break;
    case mjGEOM_ELLIPSOID:
      // The polar coordinate is acos(z/radius), so the radius must be the
      // object's z semi-axis for it to span the full 0..pi.
      tg.kind = TexGen::Kind::Spherical;
      tg.radius = sz > 0.0f ? sz : 1.0f;
      break;
    case mjGEOM_CAPSULE:
    case mjGEOM_CYLINDER:
      tg.kind = TexGen::Kind::Cylindrical;
      tg.height = sy > 0.0f ? 2.0f * sy : 1.0f;
      break;
    default:
      // Box and anything else: project along the object z axis, as the classic
      // renderer does for every 2D-textured geom without UVs.
      tg.kind = TexGen::Kind::Planar;
      tg.extent = {tile(rep0, sx), tile(rep1, sy)};
      break;
  }
  return tg;
}

// MuJoCo marks a KTX/compressed payload by storing it as a single-channel,
// single-row blob (the same test the Filament renderer uses to select
// mjPIXEL_FORMAT_KTX, model_objects.cc:501); only that renderer decodes them.
bool IsEncodedTexture(const mjModel* m, int texid) {
  const int nchannel = m->tex_nchannel[texid];
  if (nchannel != 1 && nchannel != 3 && nchannel != 4) return true;
  return nchannel == 1 && m->tex_height[texid] == 1;
}

// A host-staged RGBA8 image. A cube texture holds its six faces stacked in GL
// order (+X,-X,+Y,-Y,+Z,-Z) -- both MuJoCo's own layout and Metal's cube slice
// order -- and `width == height` is then one face's edge length.
struct TextureImage {
  std::vector<uint8_t> pixels;
  unsigned width = 0;
  unsigned height = 0;
  bool cube = false;

  bool empty() const { return pixels.empty(); }
  size_t face_bytes() const {
    return static_cast<size_t>(width) * height * 4;
  }
};

// Expands mjModel texture `texid` into tightly packed RGBA8. Single-channel
// data broadcasts to grey (MuJoCo stores roughness, metallic and occlusion maps
// that way) and three-channel data gets an opaque alpha. A square
// (height == width) cube texture is a single face repeated on all six, as the
// classic renderer treats it (render_context.c:1525).
TextureImage ReadTexture(const mjModel* m, int texid) {
  TextureImage img;
  const int nchannel = m->tex_nchannel[texid];
  const unsigned w = m->tex_width[texid];
  const unsigned h = m->tex_height[texid];
  img.cube = m->tex_type[texid] != mjTEXTURE_2D;
  img.width = w;
  img.height = img.cube ? w : h;
  const unsigned faces = img.cube ? 6 : 1;
  const bool repeat_face = img.cube && h == w;
  const mjtByte* src = m->tex_data + m->tex_adr[texid];
  const size_t texels = static_cast<size_t>(img.width) * img.height;
  img.pixels.resize(texels * 4 * faces);
  uint8_t* dst = img.pixels.data();
  for (unsigned f = 0; f < faces; ++f) {
    const mjtByte* face =
        src + (repeat_face ? 0 : static_cast<size_t>(f) * texels * nchannel);
    for (size_t t = 0; t < texels; ++t) {
      const mjtByte* p = face + t * nchannel;
      dst[0] = p[0];
      dst[1] = nchannel > 1 ? p[1] : p[0];
      dst[2] = nchannel > 2 ? p[2] : p[0];
      dst[3] = nchannel > 3 ? p[3] : 255;
      dst += 4;
    }
  }
  return img;
}

// True when two staged images can be combined channel-wise.
bool SameShape(const TextureImage& a, const TextureImage& b) {
  return a.width == b.width && a.height == b.height && a.cube == b.cube;
}

// How a staged host image is assembled from mjModel's per-role textures. Two
// roles can name the same mjModel texture yet need different packing, so the
// key carries the packing alongside the source ids. The packing also decides
// whether the sampler sRGB-decodes: MuJoCo's colour space is a property of the
// file, and a normal map stored in an sRGB-tagged PNG must still be read
// linearly.
enum class ImagePacking {
  Color,      // `primary`, sRGB-decoded if the texture declares that space
  Linear,     // `primary`, always read linearly (normal maps)
  BaseColor,  // Color, plus `secondary` (OPACITY) folded into the alpha
  Orm,        // `primary` (ORM), plus `secondary` (ROUGHNESS) into G and
              // `tertiary` (METALLIC) into B; always linear
};

struct ImageKey {
  ImagePacking packing = ImagePacking::Color;
  int primary = -1;
  int secondary = -1;
  int tertiary = -1;
  // Orm only. bourbon's glTFPbrPattern REPLACES the metallic and roughness
  // uniforms with the map rather than multiplying them, whereas glTF (and
  // MuJoCo, and the Filament renderer) treat the scalars as factors over the
  // texture. So the scalars are baked into the staged image instead, and these
  // are how: each byte scales its channel's texture (255 == x1.0), or stands in
  // for the channel outright when that channel has no texture.
  uint8_t mul_g = 255;  // roughness
  uint8_t mul_b = 0;    // metallic

  bool bound() const {
    return primary >= 0 || secondary >= 0 || tertiary >= 0;
  }
  auto operator<=>(const ImageKey&) const = default;
};

// Quantizes a [0,1] scalar to the 8-bit channel value a synthesized ORM image
// carries for it.
uint8_t ToByte(float v) {
  return static_cast<uint8_t>(std::lround(Clamp01(v) * 255.0f));
}

// Assembles the host image for `key`. Combining roles requires matching
// dimensions -- MuJoCo imposes no such constraint, so a mismatch warns and the
// offending contribution is dropped rather than sampled out of bounds.
TextureImage StageImage(const mjModel* m, const ImageKey& key) {
  switch (key.packing) {
    case ImagePacking::Color:
    case ImagePacking::Linear:
      return key.primary >= 0 ? ReadTexture(m, key.primary) : TextureImage{};

    case ImagePacking::BaseColor: {
      TextureImage base = ReadTexture(m, key.primary);
      if (key.secondary < 0) return base;
      // glTFPbrPattern has no opacity slot: baseColorMap's alpha is the only
      // place mjTEXROLE_OPACITY can land.
      const TextureImage opacity = ReadTexture(m, key.secondary);
      if (!SameShape(base, opacity)) {
        mju_warning("bourbon: opacity texture %d (%ux%u) does not match base "
                    "color texture %d (%ux%u); ignoring it",
                    key.secondary, opacity.width, opacity.height, key.primary,
                    base.width, base.height);
        return base;
      }
      for (size_t t = 0; t * 4 < base.pixels.size(); ++t) {
        base.pixels[t * 4 + 3] = opacity.pixels[t * 4];
      }
      return base;
    }

    case ImagePacking::Orm: {
      // glTF ORM packing: R = occlusion, G = roughness, B = metallic, and the
      // pattern reads .bg, i.e. (metallic, roughness). Occlusion is left at 255
      // because the pattern has no occlusion input to route it to.
      //
      // A dedicated ROUGHNESS or METALLIC texture wins over an ORM texture's
      // corresponding channel, and mul_g/mul_b apply the material's scalars
      // (see ImageKey) since the shader will not.
      const TextureImage orm_src =
          key.primary >= 0 ? ReadTexture(m, key.primary) : TextureImage{};
      const TextureImage rough_src =
          key.secondary >= 0 ? ReadTexture(m, key.secondary) : TextureImage{};
      const TextureImage metal_src =
          key.tertiary >= 0 ? ReadTexture(m, key.tertiary) : TextureImage{};

      TextureImage out;
      for (const TextureImage* c : {&orm_src, &rough_src, &metal_src}) {
        if (!c->empty()) {
          out = *c;
          break;
        }
      }
      if (out.empty()) return {};

      // A standalone roughness/metallic map is greyscale, so read its red;
      // an ORM texture carries the value in its own glTF channel.
      const TextureImage* rough =
          !rough_src.empty() ? &rough_src
                             : (!orm_src.empty() ? &orm_src : nullptr);
      const int rough_channel = !rough_src.empty() ? 0 : 1;
      const TextureImage* metal =
          !metal_src.empty() ? &metal_src
                             : (!orm_src.empty() ? &orm_src : nullptr);
      const int metal_channel = !metal_src.empty() ? 0 : 2;

      auto compatible = [&](const TextureImage* src, const char* what) {
        if (!src || SameShape(out, *src)) return src;
        mju_warning("bourbon: %s texture (%ux%u) does not match the rest of "
                    "the material's roughness/metallic set (%ux%u); ignoring it",
                    what, src->width, src->height, out.width, out.height);
        return static_cast<const TextureImage*>(nullptr);
      };
      rough = compatible(rough, "roughness");
      metal = compatible(metal, "metallic");

      for (size_t t = 0, n = out.pixels.size() / 4; t < n; ++t) {
        const unsigned g =
            rough ? rough->pixels[t * 4 + rough_channel] * key.mul_g / 255
                  : key.mul_g;
        const unsigned b =
            metal ? metal->pixels[t * 4 + metal_channel] * key.mul_b / 255
                  : key.mul_b;
        out.pixels[t * 4 + 0] = 255;
        out.pixels[t * 4 + 1] = static_cast<uint8_t>(g);
        out.pixels[t * 4 + 2] = static_cast<uint8_t>(b);
        out.pixels[t * 4 + 3] = 255;
      }
      return out;
    }
  }
  return {};
}

// The mjModel texture whose declared colour space governs a staged image, or
// -1 when the data must be read linearly regardless of what the texture says.
int ColorSpaceSource(const ImageKey& key) {
  switch (key.packing) {
    case ImagePacking::Color:
    case ImagePacking::BaseColor:
      return key.primary;
    case ImagePacking::Linear:
    case ImagePacking::Orm:
      return -1;
  }
  return -1;
}

// The complete glTF-PBR parameter set for one geom, as handed to
// Pattern::Create.
struct MaterialSpec {
  Eigen::Array4f base_color = {1.0f, 1.0f, 1.0f, 1.0f};
  // rgb = specular tint, a = specular weight. glTFPbrPattern splits the vector
  // exactly that way (outputs.specular = specular.a, specular_color =
  // specular.rgb), and the BXDF forms F0 = min(specular_color * f0(ior), 1) *
  // specular.
  Eigen::Array4f specular = {1.0f, 1.0f, 1.0f, 1.0f};
  Eigen::Array4f emissive = {0.0f, 0.0f, 0.0f, 0.0f};
  float metallic = 0.0f;
  float roughness = 1.0f;
  float ior = kDefaultIor;
  // Surface opacity. Ignored by the BXDF while alpha_mode is OPAQUE (the
  // gltf_pbr graph forces opacity to 1 in that mode); transparency support
  // sets alpha_mode and the matching SemiTransparencyKind together.
  float alpha = 1.0f;

  // glTFPbrPattern's texture slots, and the coordinate generation every one of
  // them shares (it is a property of the geom, not of the role).
  ImageKey base_color_map;
  ImageKey metallic_roughness_map;
  ImageKey normal_map;
  ImageKey emissive_map;
  TexGen texgen;

  // Quantized identity, so geoms whose materials would render identically
  // share one Pattern/Material. 1/4096 is far below any visible difference,
  // and an integer key hashes and orders exactly.
  struct Key {
    std::array<int32_t, 16> scalars{};
    std::array<ImageKey, 4> maps{};
    std::array<int32_t, 12> texgen{};
    // Two materials over the same parameters but different blend behaviour are
    // distinct Patterns, so the kind is part of the identity.
    int semi_transparency = 0;
    auto operator<=>(const Key&) const = default;
  };

  Key key(bourbon::SemiTransparencyKind semi_transparency) const {
    auto q = [](float v) {
      return static_cast<int32_t>(std::lround(v * 4096.0f));
    };
    return Key{
        {q(base_color[0]), q(base_color[1]), q(base_color[2]), q(base_color[3]),
         q(specular[0]), q(specular[1]), q(specular[2]), q(specular[3]),
         q(emissive[0]), q(emissive[1]), q(emissive[2]), q(emissive[3]),
         q(metallic), q(roughness), q(ior), q(alpha)},
        {base_color_map, metallic_roughness_map, normal_map, emissive_map},
        texgen.Key(),
        static_cast<int>(semi_transparency)};
  }
};

// Converts an OpenGL Blinn-Phong shininess (MuJoCo's mat_shininess, in [0,1],
// scaling a specular exponent up to 128) to a GGX roughness. Used only when the
// material states no explicit roughness.
float RoughnessFromShininess(float shininess) {
  const float exponent = Clamp01(shininess) * kMaxShininessExponent;
  return Clamp01(std::pow(2.0f / (exponent + 2.0f), 0.25f));
}

// Derives the glTF-PBR parameters common to geoms, flexes and skins: the scalar
// mapping and the texture-role -> pattern-slot bindings. `rgba` is the already-
// resolved effective surface colour; `matid` its material (-1 for none);
// `allow_textures` whether texture roles apply at all (a flex/skin with no
// texcoords renders untextured, matching classic). When any texture binds,
// `texgen` is stored so the sampler has coordinates.
//
// This is the one place a retained renderer reading mjModel does better than
// any mjvGeom-based one: mat_metallic and mat_roughness exist in mjModel and
// are never copied into mjvGeom, so the classic and Filament paths cannot see
// them. Both default to -1 meaning "unset" (mjs_defaultMaterial), so the
// Blinn-Phong conversion remains the fallback for a surface with no material AND
// for a material that authors shininess but no roughness.
void ApplyMaterial(MaterialSpec& spec, const mjModel* m, int matid,
                   const float rgba[4], bool allow_textures,
                   const TexGen& texgen) {
  spec.base_color = {rgba[0], rgba[1], rgba[2], rgba[3]};
  spec.alpha = rgba[3];

  const float emission =
      matid >= 0 ? m->mat_emission[matid] : kDefaultEmission;
  const float specular =
      matid >= 0 ? m->mat_specular[matid] : kDefaultSpecular;
  const float shininess =
      matid >= 0 ? m->mat_shininess[matid] : kDefaultShininess;

  const float metallic = matid >= 0 ? m->mat_metallic[matid] : -1.0f;
  spec.metallic = metallic >= 0.0f ? Clamp01(metallic) : 0.0f;

  const float roughness = matid >= 0 ? m->mat_roughness[matid] : -1.0f;
  spec.roughness = roughness >= 0.0f ? Clamp01(roughness)
                                     : RoughnessFromShininess(shininess);

  const float s = Clamp01(specular) * kSpecularToGltf;
  spec.specular = {s, s, s, 1.0f};
  spec.ior = kDefaultIor;
  // MuJoCo's emission is a scalar multiplier on the surface's own colour.
  spec.emissive = {emission * rgba[0], emission * rgba[1], emission * rgba[2],
                   0.0f};

  if (matid < 0 || !allow_textures) return;

  // Texture roles -> glTFPbrPattern slots. mat_texid is nmat x mjNTEXROLE.
  //
  //   RGB / RGBA           -> baseColorMap
  //   OPACITY              -> folded into baseColorMap's alpha (no opacity slot)
  //   ORM                  -> metallicRoughnessMap (glTF-native packing)
  //   ROUGHNESS / METALLIC -> packed into a synthesized ORM image
  //   NORMAL               -> normalMap
  //   EMISSIVE             -> emissiveMap
  //   OCCLUSION            -> dropped: the pattern has no occlusion input
  //   USER                 -> ignored
  const int* roles = m->mat_texid + matid * mjNTEXROLE;
  auto usable = [&](int role) {
    const int texid = roles[role];
    // Encoded payloads are reported once per model in BuildScene, not here.
    return (texid >= 0 && !IsEncodedTexture(m, texid)) ? texid : -1;
  };

  const int rgba_role = usable(mjTEXROLE_RGBA);
  const int base = rgba_role >= 0 ? rgba_role : usable(mjTEXROLE_RGB);
  if (base >= 0) {
    spec.base_color_map = {ImagePacking::BaseColor, base,
                           usable(mjTEXROLE_OPACITY)};
  }

  const int orm = usable(mjTEXROLE_ORM);
  const int rough_tex = usable(mjTEXROLE_ROUGHNESS);
  const int metal_tex = usable(mjTEXROLE_METALLIC);
  if (orm >= 0 || rough_tex >= 0 || metal_tex >= 0) {
    // A channel that has a texture takes the scalar as a glTF-style factor
    // over it, which is 1 when the material leaves it unset; a channel with no
    // texture takes the scalar outright, which is the same fallback the
    // uniform-only path uses (Blinn-Phong roughness, non-metal).
    const bool has_rough = orm >= 0 || rough_tex >= 0;
    const bool has_metal = orm >= 0 || metal_tex >= 0;
    const uint8_t mul_g =
        has_rough ? ToByte(roughness >= 0.0f ? roughness : 1.0f)
                  : ToByte(spec.roughness);
    const uint8_t mul_b = has_metal
                              ? ToByte(metallic >= 0.0f ? metallic : 1.0f)
                              : ToByte(spec.metallic);
    spec.metallic_roughness_map = {ImagePacking::Orm, orm,   rough_tex,
                                   metal_tex,         mul_g, mul_b};
    // The scalars now live entirely in the staged image, and the shader reads
    // the map instead of the uniforms. Neutralize them so two materials that
    // differ only in a scalar the map already carries still share a Pattern.
    spec.metallic = 0.0f;
    spec.roughness = 0.0f;
  }

  const int normal = usable(mjTEXROLE_NORMAL);
  if (normal >= 0) spec.normal_map = {ImagePacking::Linear, normal};

  const int emissive = usable(mjTEXROLE_EMISSIVE);
  if (emissive >= 0) {
    spec.emissive_map = {ImagePacking::Color, emissive};
    // The emissive map multiplies the emissive uniform, so a material with a
    // map but the default emission of 0 would be black. Use the surface colour
    // at full strength in that case and let the map carry the variation.
    if (emission <= 0.0f) {
      spec.emissive = {rgba[0], rgba[1], rgba[2], 0.0f};
    }
  }

  // Only give the spec a coordinate generation if something actually samples
  // with it: texgen depends on the surface's own size, so letting it into the
  // key for an untextured material would fragment the cache by size.
  if (spec.base_color_map.bound() || spec.metallic_roughness_map.bound() ||
      spec.normal_map.bound() || spec.emissive_map.bound()) {
    spec.texgen = texgen;
  }
}

// Derives the glTF-PBR parameters for geom `i`.
MaterialSpec MaterialSpecForGeom(const mjModel* m, int i) {
  MaterialSpec spec;
  float rgba[4];
  EffectiveGeomRgba(m, i, rgba);
  ApplyMaterial(spec, m, m->geom_matid[i], rgba, /*allow_textures=*/true,
                TexGenForGeom(m, i));
  return spec;
}

// The coordinate generation for a flex/skin surface. Unlike SDF primitives,
// flexes and skins carry real per-vertex texture coordinates (flextexcoord /
// skin_texcoord), so the shape's own st is used directly, with the material's
// texrepeat applied as a scale (TransformUVCoordMap, the KHR_texture_transform
// path -- the same as a mesh with UVs).
TexGen FlexSkinTexGen(const mjModel* m, int matid) {
  TexGen tg;
  tg.kind = TexGen::Kind::Uv;
  const float rep0 = matid >= 0 ? m->mat_texrepeat[2 * matid + 0] : 1.0f;
  const float rep1 = matid >= 0 ? m->mat_texrepeat[2 * matid + 1] : 1.0f;
  tg.scale = {rep0 > 0.0f ? rep0 : 1.0f, rep1 > 0.0f ? rep1 : 1.0f};
  return tg;
}

// glTF-PBR parameters for flex `f`. Textures apply only when the flex has
// texcoords (matching addFlexGeoms, which strips the material otherwise).
MaterialSpec MaterialSpecForFlex(const mjModel* m, int f) {
  MaterialSpec spec;
  const int matid = m->flex_matid[f];
  float rgba[4];
  EffectiveRgba(m, matid, m->flex_rgba + 4 * f, rgba);
  ApplyMaterial(spec, m, matid, rgba, m->flex_texcoordadr[f] >= 0,
                FlexSkinTexGen(m, matid));
  return spec;
}

// glTF-PBR parameters for skin `s`.
MaterialSpec MaterialSpecForSkin(const mjModel* m, int s) {
  MaterialSpec spec;
  const int matid = m->skin_matid[s];
  float rgba[4];
  EffectiveRgba(m, matid, m->skin_rgba + 4 * s, rgba);
  ApplyMaterial(spec, m, matid, rgba, m->skin_texcoordadr[s] >= 0,
                FlexSkinTexGen(m, matid));
  return spec;
}

// The alpha geom `i` renders with. mjVIS_TRANSPARENT fades dynamic-category
// geoms by vis.map.alpha, exactly as setMaterial does
// (engine_vis_visualize.c:251); a body is static, and so exempt, when it is
// welded to the world (bodycategory, :156).
bool IsDynamicGeom(const mjModel* m, int i) {
  return m->body_weldid[m->geom_bodyid[i]] != 0;
}

// Whether model geom `i` should be drawn under `opt`, mirroring the two model-
// geom filters mjv_addGeoms applies (engine_vis_visualize.c:962-971) and the
// filter Pick passes to mj_ray (interaction.cc:347), so render and pick agree:
// its geom group must be enabled, and static-body geoms are hidden unless
// mjVIS_STATIC is set. All other mjVIS_* flags act on categories bourbon does
// not yet retain (sites/tendons/actuators) or are handled elsewhere
// (mjVIS_TRANSPARENT in UpdateTransparency).
bool GeomVisibleUnderOption(const mjModel* m, const mjvOption& opt, int i) {
  const int g = m->geom_group[i];
  const int gc = g < 0 ? 0 : (g >= mjNGROUP ? mjNGROUP - 1 : g);
  if (!opt.geomgroup[gc]) return false;
  if (!opt.flags[mjVIS_STATIC] && m->body_weldid[m->geom_bodyid[i]] == 0) {
    return false;
  }
  return true;
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
  // Owned by `node`; held so the material can be swapped (a ShapeInstance's
  // material is bound at addShape and has no setter, so a swap is a
  // removeSGObject + addShape pair).
  bourbon::ShapeInstance* shape_instance = nullptr;
  // Whether this geom currently shows the selection glow. While true, its
  // ShapeInstance holds a glow variant of its material (its own spec plus an
  // emissive boost) instead of `material`, and the transparency sync leaves the
  // instance alone. A translucent overlay would be the tidier design, but
  // bourbon's OIT is LDR while the scene is HDR/photometric, so a blended
  // overlay collapses to a dark smudge under the exposure -- an opaque emissive
  // boost on the body's own material (as the classic renderer does) is the
  // approach that actually reads.
  bool highlighted = false;

  // Transparency state. A Pattern's SemiTransparencyKind is fixed at
  // construction (Pattern only exposes getSemiTransparency()), so a geom that
  // becomes translucent needs a second, Blend-built material rather than a
  // mutated one. `spec` is the geom's parameter set at its authored (unfaded)
  // alpha; `applied_alpha` and `blended` describe the material currently bound.
  MaterialSpec spec;
  bool dynamic = false;  // mjCAT_DYNAMIC: mjVIS_TRANSPARENT fades this geom
  bool blended = false;
  float applied_alpha = 1.0f;
  // Whether the node is currently shown. Driven by the mjvOption geom-group and
  // mjVIS_STATIC flags (UpdateVisibility); tracked so a node is only touched
  // when its computed visibility actually changes. Nodes are created visible.
  bool visible = true;
};

// A retained flex or skin surface. Unlike a geom, its vertices are world-space
// and deform every frame, so the node and its identity MatrixTransformer are
// permanent while the Shape is swapped out whenever the geometry changes: the
// swap is gated on a checksum of the source vertices, so a static flex/skin
// costs only the checksum after the first build. All the swap goes through one
// function (SetFlexSkinShape) so a future in-place TriangleMesh position update
// (a planned bourbon-side change) is a localized edit. Flexes and skins share
// this type; they live in parallel lists indexed by flex/skin id.
struct FlexSkin {
  bourbon::SGNode* node = nullptr;  // owned by world->model(), identity xform
  bourbon::RefPtr<bourbon::MatrixTransformer> xform;  // identity (world-space verts)
  bourbon::RefPtr<bourbon::Shape> shape;              // swapped on deformation
  bourbon::ShapeInstance* shape_instance = nullptr;   // owned by node
  bourbon::RefPtr<bourbon::Material> material;         // currently-bound material
  MaterialSpec spec;         // base parameter set (before alpha fade / glow)
  bool has_shape = false;    // whether a Shape is currently attached
  bool blended = false;      // current material is a Blend variant
  bool highlighted = false;  // current material carries the selection glow
  bool visible = true;       // node visibility (flags/group)
  uint64_t checksum = 0;     // change-detection over the source vertices
};

// FNV-1a over `n` 32-bit words (floats reinterpreted), mixed with `seed`. Used
// to detect whether a flex/skin's source vertices changed since the last frame
// so an unchanged surface skips the Shape rebuild entirely.
uint64_t HashFloats(const float* p, size_t n, uint64_t seed) {
  uint64_t h = (1469598103934665603ull ^ seed) * 1099511628211ull;
  const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
  for (size_t i = 0; i < n; ++i) {
    h ^= w[i];
    h *= 1099511628211ull;
  }
  return h;
}

// A de-indexed triangle soup (3 vertices per triangle) staged on the host
// before upload as a bourbon TriangleMesh. Normals are always supplied;
// tangents/bitangents are derived at upload; texcoords are optional.
struct MeshBuild {
  std::vector<Eigen::Vector3f> positions;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector2f> sts;
  bool has_st = false;
};

// Frisvad's branchless orthonormal basis: given a unit normal `n`, produces a
// tangent `r` and bitangent `u`. Ported from the prior effort's computeTangents.
void ComputeTangents(const Eigen::Vector3f& n, Eigen::Vector3f& r,
                     Eigen::Vector3f& u) {
  const float sign = n.z() < 0.0f ? -1.0f : 1.0f;
  const float a = -1.0f / (sign + n.z());
  const float b = n.x() * n.y() * a;
  r = Eigen::Vector3f(1.0f + sign * n.x() * n.x() * a, sign * b,
                      -sign * n.x());
  u = Eigen::Vector3f(b, sign + n.y() * n.y() * a, -n.y());
}

// Unit face normal of triangle (a,b,c), CCW; falls back to +Z if degenerate.
Eigen::Vector3f FaceNormal(const Eigen::Vector3f& a, const Eigen::Vector3f& b,
                           const Eigen::Vector3f& c) {
  Eigen::Vector3f fn = (b - a).cross(c - a);
  const float len = fn.norm();
  return len > 1e-12f ? Eigen::Vector3f(fn / len)
                      : Eigen::Vector3f(0.0f, 0.0f, 1.0f);
}

// Appends a flat-shaded triangle (one face normal shared by its 3 vertices).
void AppendTri(MeshBuild& mb, const Eigen::Vector3f& a, const Eigen::Vector3f& b,
               const Eigen::Vector3f& c, const Eigen::Vector2f& ua,
               const Eigen::Vector2f& ub, const Eigen::Vector2f& uc) {
  const Eigen::Vector3f fn = FaceNormal(a, b, c);
  mb.positions.push_back(a); mb.normals.push_back(fn); mb.sts.push_back(ua);
  mb.positions.push_back(b); mb.normals.push_back(fn); mb.sts.push_back(ub);
  mb.positions.push_back(c); mb.normals.push_back(fn); mb.sts.push_back(uc);
}

// Appends a flat-shaded quad as two triangles (a,b,d)+(d,b,c), matching the
// Filament renderer's winding (model_objects.cc append_quad).
void AppendQuad(MeshBuild& mb, const Eigen::Vector3f& a,
                const Eigen::Vector3f& b, const Eigen::Vector3f& c,
                const Eigen::Vector3f& d, const Eigen::Vector2f& ua,
                const Eigen::Vector2f& ub, const Eigen::Vector2f& uc,
                const Eigen::Vector2f& ud) {
  AppendTri(mb, a, b, d, ua, ub, ud);
  AppendTri(mb, d, b, c, ud, ub, uc);
}

// De-indexes mesh `meshid` into a triangle soup. MuJoCo carries three
// independent index streams (mesh_face, mesh_facenormal, mesh_facetexcoord), so
// a single vertex buffer requires walking faces and emitting unique triples.
// Applies the hard-normal heuristic (substitute the face normal when the stored
// vertex normal deviates too far, keeping sharp edges crisp).
MeshBuild BuildMeshGeometry(const mjModel* m, int meshid) {
  MeshBuild mb;
  const int vertadr = m->mesh_vertadr[meshid];
  const int normaladr = m->mesh_normaladr[meshid];
  const int texcoordadr = m->mesh_texcoordadr[meshid];
  const int faceadr = m->mesh_faceadr[meshid];
  const int facenum = m->mesh_facenum[meshid];
  mb.has_st = texcoordadr >= 0;
  const size_t vcount = static_cast<size_t>(facenum) * 3;
  mb.positions.reserve(vcount);
  mb.normals.reserve(vcount);
  if (mb.has_st) mb.sts.reserve(vcount);

  for (int f = faceadr; f < faceadr + facenum; ++f) {
    const int* face = m->mesh_face + 3 * f;
    const Eigen::Vector3f p[3] = {
        Eigen::Vector3f(m->mesh_vert[3 * (face[0] + vertadr) + 0],
                        m->mesh_vert[3 * (face[0] + vertadr) + 1],
                        m->mesh_vert[3 * (face[0] + vertadr) + 2]),
        Eigen::Vector3f(m->mesh_vert[3 * (face[1] + vertadr) + 0],
                        m->mesh_vert[3 * (face[1] + vertadr) + 1],
                        m->mesh_vert[3 * (face[1] + vertadr) + 2]),
        Eigen::Vector3f(m->mesh_vert[3 * (face[2] + vertadr) + 0],
                        m->mesh_vert[3 * (face[2] + vertadr) + 1],
                        m->mesh_vert[3 * (face[2] + vertadr) + 2])};
    const Eigen::Vector3f fn = FaceNormal(p[0], p[1], p[2]);
    const int* fnorm = m->mesh_facenormal + 3 * f;
    const int* ftex = mb.has_st ? m->mesh_facetexcoord + 3 * f : nullptr;
    for (int k = 0; k < 3; ++k) {
      mb.positions.push_back(p[k]);
      const float* n = m->mesh_normal + 3 * (fnorm[k] + normaladr);
      Eigen::Vector3f vn(n[0], n[1], n[2]);
      if (vn.dot(fn) < 0.8f) vn = fn;  // hard-normal heuristic
      mb.normals.push_back(vn);
      if (mb.has_st) {
        const float* t = m->mesh_texcoord + 2 * (ftex[k] + texcoordadr);
        mb.sts.emplace_back(t[0], t[1]);
      }
    }
  }
  return mb;
}

// Builds a plane geom as a subdivided grid in the geom-local XY plane (+Z
// normal), with its extent baked into the vertices (so the node transform stays
// rigid, matching the per-frame update). Infinite dimensions (size == 0) size to
// the far-plane distance. The grid subdivision is essential: a single quad
// spanning an "infinite" plane (tens of scene-extents) is one enormous triangle
// pair, which stresses clip/guard-band and depth interpolation and renders
// incorrectly (the classic and prior bourbon renderers subdivide for the same
// reason). Cell size targets the scene extent; texture-repeat UVs land in
// Stage 3.
MeshBuild BuildPlaneGeometry(const mjModel* m, int geomid) {
  MeshBuild mb;
  mb.has_st = true;
  const mjtNum* size = m->geom_size + 3 * geomid;
  const float far = static_cast<float>(m->vis.map.zfar * m->stat.extent);
  const float hx = size[0] > 0 ? static_cast<float>(size[0]) : far;
  const float hy = size[1] > 0 ? static_cast<float>(size[1]) : far;
  const float cell = std::max(static_cast<float>(m->stat.extent), 1e-3f);
  auto subdiv = [&](float half) {
    int n = static_cast<int>(std::ceil(2.0f * half / cell));
    if (n < m->vis.quality.numquads) n = m->vis.quality.numquads;
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    return n;
  };
  const int nx = subdiv(hx), ny = subdiv(hy);
  // UVs follow the classic renderer's plane exactly (render_context.c:205-220):
  // a finite dimension spans 0..1 across the plane, an infinite one is measured
  // in world units so that the texrepeat scale is absolute; v runs opposite to
  // +y in both cases.
  auto plane_u = [&](float x) {
    return size[0] > 0 ? (x + hx) / (2.0f * hx) : 0.5f * x;
  };
  auto plane_v = [&](float y) {
    return size[1] > 0 ? 1.0f - (y + hy) / (2.0f * hy) : -0.5f * y;
  };
  for (int ix = 0; ix < nx; ++ix) {
    const float x0 = -hx + 2.0f * hx * ix / nx;
    const float x1 = -hx + 2.0f * hx * (ix + 1) / nx;
    const float u0 = plane_u(x0);
    const float u1 = plane_u(x1);
    for (int iy = 0; iy < ny; ++iy) {
      const float y0 = -hy + 2.0f * hy * iy / ny;
      const float y1 = -hy + 2.0f * hy * (iy + 1) / ny;
      const float v0 = plane_v(y0);
      const float v1 = plane_v(y1);
      AppendQuad(mb, Eigen::Vector3f(x0, y0, 0.0f),
                 Eigen::Vector3f(x1, y0, 0.0f), Eigen::Vector3f(x1, y1, 0.0f),
                 Eigen::Vector3f(x0, y1, 0.0f), Eigen::Vector2f(u0, v0),
                 Eigen::Vector2f(u1, v0), Eigen::Vector2f(u1, v1),
                 Eigen::Vector2f(u0, v1));
    }
  }
  return mb;
}

// Tessellates height field `hid` into a closed box: a top surface (4 triangles
// per grid cell around a centre vertex, avoiding spurious bumps), four skirts
// down to the base, and a base plane. Flat per-triangle normals. Ported from
// the Filament renderer (model_objects.cc FillHeightFieldBuffer).
MeshBuild BuildHfieldGeometry(const mjModel* m, int hid) {
  MeshBuild mb;
  mb.has_st = true;
  const float* data = m->hfield_data + m->hfield_adr[hid];
  const int nrow = m->hfield_nrow[hid];
  const int ncol = m->hfield_ncol[hid];
  const float fheight = 0.5f * (nrow - 1);
  const float fwidth = 0.5f * (ncol - 1);
  float sz[4];
  for (int i = 0; i < 4; ++i) {
    sz[i] = static_cast<float>(m->hfield_size[4 * hid + i]);
  }
  auto pos = [&](int r, int c) {
    return Eigen::Vector3f(sz[0] * (c / fwidth - 1.0f),
                           sz[1] * (r / fheight - 1.0f),
                           sz[2] * data[r * ncol + c]);
  };
  auto uv = [&](int r, int c) {
    return Eigen::Vector2f(static_cast<float>(c) / (ncol - 1),
                           1.0f - static_cast<float>(r) / (nrow - 1));
  };

  // Top surface.
  for (int row = 0; row < nrow - 1; ++row) {
    for (int col = 0; col < ncol - 1; ++col) {
      const Eigen::Vector3f a = pos(row, col), b = pos(row, col + 1),
                            c = pos(row + 1, col + 1), d = pos(row + 1, col);
      float mid_z;
      if (a.z() == c.z() && b.z() != d.z()) {
        mid_z = a.z();
      } else if (a.z() != c.z() && b.z() == d.z()) {
        mid_z = b.z();
      } else {
        mid_z = std::max((a.z() + c.z()) * 0.5f, (b.z() + d.z()) * 0.5f);
      }
      const Eigen::Vector3f mid((a.x() + b.x()) * 0.5f, (a.y() + d.y()) * 0.5f,
                                mid_z);
      const Eigen::Vector2f ua = uv(row, col), ub = uv(row, col + 1),
                            uc = uv(row + 1, col + 1), ud = uv(row + 1, col);
      const Eigen::Vector2f umid(
          static_cast<float>(col + 0.5f) / (ncol - 1),
          1.0f - static_cast<float>(row + 0.5f) / (nrow - 1));
      AppendTri(mb, a, b, mid, ua, ub, umid);
      AppendTri(mb, b, c, mid, ub, uc, umid);
      AppendTri(mb, c, d, mid, uc, ud, umid);
      AppendTri(mb, d, a, mid, ud, ua, umid);
    }
  }
  // Left / right skirts.
  for (int row = 0; row < nrow - 1; ++row) {
    const Eigen::Vector3f a = pos(row, 0), b = pos(row + 1, 0);
    AppendQuad(mb, a, b, Eigen::Vector3f(b.x(), b.y(), -sz[3]),
               Eigen::Vector3f(a.x(), a.y(), -sz[3]),
               Eigen::Vector2f(0.0f, 1.0f - static_cast<float>(row) / (nrow - 1)),
               Eigen::Vector2f(0.0f, 1.0f - static_cast<float>(row + 1) / (nrow - 1)),
               Eigen::Vector2f(0.0f, 1.0f - static_cast<float>(row + 1) / (nrow - 1)),
               Eigen::Vector2f(0.0f, 1.0f - static_cast<float>(row) / (nrow - 1)));
  }
  for (int row = 0; row < nrow - 1; ++row) {
    const Eigen::Vector3f a = pos(row + 1, ncol - 1), b = pos(row, ncol - 1);
    AppendQuad(mb, a, b, Eigen::Vector3f(b.x(), b.y(), -sz[3]),
               Eigen::Vector3f(a.x(), a.y(), -sz[3]),
               Eigen::Vector2f(1.0f, 1.0f - static_cast<float>(row + 1) / (nrow - 1)),
               Eigen::Vector2f(1.0f, 1.0f - static_cast<float>(row) / (nrow - 1)),
               Eigen::Vector2f(1.0f, 1.0f - static_cast<float>(row) / (nrow - 1)),
               Eigen::Vector2f(1.0f, 1.0f - static_cast<float>(row + 1) / (nrow - 1)));
  }
  // Front / back skirts.
  for (int col = 0; col < ncol - 1; ++col) {
    const Eigen::Vector3f a = pos(0, col), dd = pos(0, col + 1);
    AppendQuad(mb, a, Eigen::Vector3f(a.x(), a.y(), -sz[3]),
               Eigen::Vector3f(dd.x(), dd.y(), -sz[3]), dd,
               Eigen::Vector2f(static_cast<float>(col) / (ncol - 1), 1.0f),
               Eigen::Vector2f(static_cast<float>(col) / (ncol - 1), 1.0f),
               Eigen::Vector2f(static_cast<float>(col + 1) / (ncol - 1), 1.0f),
               Eigen::Vector2f(static_cast<float>(col + 1) / (ncol - 1), 1.0f));
  }
  for (int col = 0; col < ncol - 1; ++col) {
    const Eigen::Vector3f a = pos(nrow - 1, col + 1), dd = pos(nrow - 1, col);
    AppendQuad(mb, a, Eigen::Vector3f(a.x(), a.y(), -sz[3]),
               Eigen::Vector3f(dd.x(), dd.y(), -sz[3]), dd,
               Eigen::Vector2f(static_cast<float>(col + 1) / (ncol - 1), 0.0f),
               Eigen::Vector2f(static_cast<float>(col + 1) / (ncol - 1), 0.0f),
               Eigen::Vector2f(static_cast<float>(col) / (ncol - 1), 0.0f),
               Eigen::Vector2f(static_cast<float>(col) / (ncol - 1), 0.0f));
  }
  // Base (sized by visualization quality, not the grid resolution).
  const float bw = 0.5f * m->vis.quality.numquads;
  const float bh = 0.5f * m->vis.quality.numquads;
  for (int row = 0; row < m->vis.quality.numquads; ++row) {
    for (int col = 0; col < m->vis.quality.numquads; ++col) {
      const float x0 = sz[0] * ((col + 0) / bw - 1.0f);
      const float x1 = sz[0] * ((col + 1) / bw - 1.0f);
      const float y0 = sz[1] * ((row + 0) / bh - 1.0f);
      const float y1 = sz[1] * ((row + 1) / bh - 1.0f);
      const Eigen::Vector2f uv0((col + 0) / (2.0f * bw),
                                1.0f - (row + 0) / (2.0f * bh));
      const Eigen::Vector2f uv1((col + 1) / (2.0f * bw),
                                1.0f - (row + 1) / (2.0f * bh));
      AppendQuad(mb, Eigen::Vector3f(x0, y0, -sz[3]),
                 Eigen::Vector3f(x0, y1, -sz[3]),
                 Eigen::Vector3f(x1, y1, -sz[3]),
                 Eigen::Vector3f(x1, y0, -sz[3]),
                 Eigen::Vector2f(uv0.x(), uv0.y()),
                 Eigen::Vector2f(uv0.x(), uv1.y()),
                 Eigen::Vector2f(uv1.x(), uv1.y()),
                 Eigen::Vector2f(uv1.x(), uv0.y()));
    }
  }
  return mb;
}

// --- Decorations -------------------------------------------------------------
//
// MuJoCo decorations (contact points/forces, joints, COM markers, perturb
// ghosts, plugin geoms, ...) arrive each frame through a private mjvScene, not
// through mjData. They are inherently dynamic -- an arrow's length tracks a
// contact force -- so the retained-mode win here is different from the model
// geoms': rather than skip unchanged transforms, we keep a grow-to-high-water
// pool of nodes that is never torn down, and drive each one's size, transform
// and colour through DGInputs. Because every SDF kind now recomputes its march
// bounds from its live DGInputs each evaluation (SDF3D.cpp), a primitive can be
// resized every frame with neither a Shape rebuild nor a clipped surface, and a
// per-part Pattern lets the colour be re-pushed without a graph mutation. So the
// steady state (a stable set of decorations) is pure setValueIfChanged.
//
// Composites (arrows) are several SDF parts under one geom frame; each part is
// its own root node with a rigid local offset baked into its world transform
// (SDF nodes must stay rigid -- R2 -- so the head cone's widening is baked into
// its radii, never a node scale). Triangles are the one rasterized decoration,
// so they may use a node scale.

// Which concrete bourbon shape backs a decoration part. The tag lets the pool
// downcast the RefPtr<Shape> to push size DGInputs for a live resize.
enum class DecorShapeKind { RoundCone, Frustum, Cube, Ellipsoid, BoxFrame,
                            Triangle, Mesh };

// One part of a decoration geom: a shape, its rigid (or, for a triangle,
// scaling) offset within the geom frame, and the size parameters to push into
// the shape's DGInputs. Interpretation of the size fields is by `kind`:
//   RoundCone : radius, height              (sphere: height 0; capsule)
//   Frustum   : base_radius, top_radius, base_z, top_z   (cylinder/shaft/cone/line)
//   Cube      : ex, ey, ez                  (full extents)
//   Ellipsoid : ex, ey, ez                  (radii)
//   BoxFrame  : ex, ey, ez, thickness       (full extents + edge-bar thickness)
//   Triangle  : `local` carries the size as a scale; no DGInputs
//   Mesh      : `mesh_dataid` selects a cached mesh shape; no DGInputs
struct DecorPartLayout {
  DecorShapeKind kind = DecorShapeKind::RoundCone;
  Eigen::Affine3f local = Eigen::Affine3f::Identity();
  float base_radius = 0, top_radius = 0, base_z = 0, top_z = 0;  // Frustum
  float radius = 0, height = 0;                                   // RoundCone
  float ex = 0, ey = 0, ez = 0;                                   // Cube/Ellipsoid/BoxFrame
  float thickness = 0;                                            // BoxFrame
  int mesh_dataid = -1;                                           // Mesh
};

// Arrow proportions, matching the classic renderer's builtin display lists
// (render_gl3.c:421-448 with the cone/cylinder extents from render_context.c:
// cone spans z in [0,1] base radius 1, cylinder spans z in [-1,1] radius 1). An
// arrow's size is {shaft_radius, shaft_radius, length}; the shaft runs to
// length/3 and the head from there to length/2, with the wedged head widened by
// kArrowHeadScale.
constexpr float kArrowHeadScale = 1.75f;

// A LINE decoration is a thin cylinder. Its width (mjvGeom.size[0]) is authored
// in screen pixels for the classic GL_LINES path, which has no object-space
// meaning here; use a small fraction of the scene extent so a line reads as a
// hairline at typical zoom. Tunable.
constexpr float kLineRadiusFraction = 0.0015f;
// LINEBOX edge-bar thickness, likewise a fraction of the scene extent.
constexpr float kLineBoxThicknessFraction = 0.0015f;

// Upper bound on the private decoration mjvScene. Contacts and force arrows can
// be numerous; this is generous enough for typical studio scenes, and the
// extra_geoms append loop caps against it.
constexpr int kDecorSceneMaxGeom = 10000;

// Selection highlight: a warm emissive glow added (in nits) to the selected
// body's own material, so the body keeps its shape/shading and just glows. An
// opaque emissive boost, not a translucent overlay, because bourbon's OIT is LDR
// under an HDR/photometric exposure (see the note on Renderable::highlighted).
constexpr float kHighlightColor[3] = {1.0f, 0.6f, 0.15f};

// Fills `out` with the part layout for decoration geom `geom` (sizes in the
// mjvGeom convention: size[2] is the capsule/cylinder half-length, NOT size[1]
// -- see engine_vis_visualize.c:327 and the plan's R4). `scene_extent` scales
// the screen-referred line widths. Returns false for a geom type this pool does
// not render (label/flex/skin/plane/hfield/sdf/none), which the caller skips.
bool LayoutDecoration(const mjvGeom& geom, float scene_extent,
                      std::vector<DecorPartLayout>& out) {
  out.clear();
  const float s0 = geom.size[0], s1 = geom.size[1], s2 = geom.size[2];
  auto frustum = [](float br, float tr, float bz, float tz) {
    DecorPartLayout p;
    p.kind = DecorShapeKind::Frustum;
    p.base_radius = br;
    p.top_radius = tr;
    p.base_z = bz;
    p.top_z = tz;
    return p;
  };
  switch (geom.type) {
    case mjGEOM_SPHERE: {
      DecorPartLayout p;
      p.kind = DecorShapeKind::RoundCone;
      p.radius = s0;
      p.height = 0.0f;
      out.push_back(p);
    } break;
    case mjGEOM_CAPSULE: {
      DecorPartLayout p;
      p.kind = DecorShapeKind::RoundCone;
      p.radius = s0;
      p.height = 2.0f * s2;  // centre-to-centre of the end caps
      out.push_back(p);
    } break;
    case mjGEOM_CYLINDER:
      out.push_back(frustum(s0, s0, -s2, s2));
      break;
    case mjGEOM_BOX: {
      DecorPartLayout p;
      p.kind = DecorShapeKind::Cube;
      p.ex = 2.0f * s0;
      p.ey = 2.0f * s1;
      p.ez = 2.0f * s2;
      out.push_back(p);
    } break;
    case mjGEOM_ELLIPSOID: {
      DecorPartLayout p;
      p.kind = DecorShapeKind::Ellipsoid;
      p.ex = s0;
      p.ey = s1;
      p.ez = s2;
      out.push_back(p);
    } break;
    case mjGEOM_MESH:
    case mjGEOM_SDF: {
      if (geom.dataid < 0) return false;
      DecorPartLayout p;
      p.kind = DecorShapeKind::Mesh;
      p.mesh_dataid = geom.dataid / 2;  // 2*id shaded / 2*id+1 hull share verts
      out.push_back(p);
    } break;
    case mjGEOM_ARROW:
      out.push_back(frustum(s0, s0, 0.0f, s2 / 3.0f));
      out.push_back(
          frustum(kArrowHeadScale * s0, 0.0f, s2 / 3.0f, s2 / 2.0f));
      break;
    case mjGEOM_ARROW1:
      out.push_back(frustum(s0, s0, 0.0f, s2 / 3.0f));
      out.push_back(frustum(s0, 0.0f, s2 / 3.0f, s2 / 2.0f));
      break;
    case mjGEOM_ARROW2:
      out.push_back(frustum(s0, s0, -s2 / 3.0f, s2 / 3.0f));
      out.push_back(
          frustum(kArrowHeadScale * s0, 0.0f, s2 / 3.0f, s2 / 2.0f));
      // The -z head is a cone whose apex (radius 0) is the more negative z.
      out.push_back(
          frustum(0.0f, kArrowHeadScale * s0, -s2 / 2.0f, -s2 / 3.0f));
      break;
    case mjGEOM_LINE: {
      const float r = std::max(kLineRadiusFraction * scene_extent, 1e-5f);
      out.push_back(frustum(r, r, 0.0f, s2));
    } break;
    case mjGEOM_LINEBOX: {
      DecorPartLayout p;
      p.kind = DecorShapeKind::BoxFrame;
      p.ex = 2.0f * s0;
      p.ey = 2.0f * s1;
      p.ez = 2.0f * s2;
      p.thickness = std::max(kLineBoxThicknessFraction * scene_extent, 1e-5f);
      out.push_back(p);
    } break;
    case mjGEOM_TRIANGLE: {
      DecorPartLayout p;
      p.kind = DecorShapeKind::Triangle;
      // A unit triangle (0,0,0)-(1,0,0)-(0,1,0) scaled to the geom's size; a
      // TriangleMesh is rasterized, so a node scale is safe here.
      Eigen::Matrix3f scale = Eigen::Matrix3f::Zero();
      scale(0, 0) = s0 > 0 ? s0 : 1.0f;
      scale(1, 1) = s1 > 0 ? s1 : 1.0f;
      scale(2, 2) = 1.0f;
      p.local = Eigen::Affine3f::Identity();
      p.local.linear() = scale;
      out.push_back(p);
    } break;
    default:
      return false;  // plane/hfield/flex/skin/label/none: not this pool
  }
  return true;
}

}  // namespace

// One realized decoration part: a root node with a rigid (or, for a triangle,
// scaling) transform, a shape whose size is pushed through DGInputs, and a
// dedicated unlit Pattern/Material whose emissive colour is re-pushed live. The
// Pattern is owned per-part precisely so the colour update is a setValue rather
// than a material swap (which would mutate the graph and re-dirty morphology).
struct DecorPart {
  bourbon::SGNode* node = nullptr;  // owned by world->model()
  bourbon::RefPtr<bourbon::MatrixTransformer> xform;
  bourbon::RefPtr<bourbon::Shape> shape;
  bourbon::RefPtr<bourbon::Pattern> pattern;
  bourbon::RefPtr<bourbon::Material> material;
  bourbon::ShapeInstance* instance = nullptr;
  DecorShapeKind kind = DecorShapeKind::RoundCone;
  bool visible = false;
  bool blended = false;
  // Last colour pushed into the pattern, so an unchanged colour is not
  // re-pushed (which would re-dirty the pattern's arguments every frame).
  std::array<int32_t, 4> rgba_key = {-1, -1, -1, -1};
};

// A decoration slot holds all the parts of one decoration geom. Slots are
// pooled by ordinal (the i-th decoration geom of the frame reuses slot i); the
// pool only ever grows, and unused parts/slots are hidden, never destroyed.
struct DecorSlot {
  std::vector<DecorPart> parts;
};

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
  // (pass chain), each with its own evaluator and scratch heap. Each heap is
  // declared -- and so destroyed -- before its evaluator: a TaskGraphEvaluator
  // can still be holding a leftover suspended GPU encode (SuspendedEncodingPool)
  // whose captured FramebufferSpec references a TextureAllocation that calls
  // back into its owning heap on destruction, so the heap must outlive the
  // evaluator.
  std::unique_ptr<bourbon::TaskGraph> model_graph;
  std::unique_ptr<bourbon::PersistentHeap> model_heap;
  std::unique_ptr<bourbon::TaskGraphEvaluator> model_graph_evaluator;

  std::unique_ptr<bourbon::TaskGraph> render_graph;
  std::unique_ptr<bourbon::PersistentHeap> render_heap;
  // Persistent heap for the pass-graph tasks' construction-time allocations
  // (integrator, ShadowTask cull buffers, render-target rings, ...). Allocated
  // once and never freed per frame, unlike render_heap (per-frame eval scratch),
  // so those buffers are not reclaimed underneath the tasks. Declared before
  // render_graph_evaluator for the same reason as render_heap above.
  std::unique_ptr<bourbon::PersistentHeap> pass_heap;
  std::unique_ptr<bourbon::TaskGraphEvaluator> render_graph_evaluator;

  std::unique_ptr<bourbon::World> world;

  // Camera + projection (live via DGInputs, updated per frame).
  bourbon::RefPtr<bourbon::MatrixProjection> projection;
  bourbon::RefPtr<bourbon::Camera> camera;
  // World->clip matrix (projection * view) for the current frame, used to place
  // geom labels in the ImGui overlay. `camera_valid` is set each frame the
  // camera block runs; a frame with no camera leaves labels un-drawn.
  Eigen::Matrix4f clip_from_world = Eigen::Matrix4f::Identity();
  bool camera_valid = false;

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

  // Structural pass-graph key: the pass graph is rebuilt when the draw-
  // submission mode or the set of shadow sources changes. Shadows are only
  // producible in a non-Direct mode (shadowDraws() is null in Direct), and the
  // shadow task bakes one encode chain per shadow source at construction, so an
  // add/remove of a shadow source (reflected in shadow_source_morphology) is
  // structural. `built_valid` is false until the first build.
  // The order-independent transparency mode is also structural: the integrator
  // captures it at construction, and with OITKind::None both the forward
  // integration and viz passes SKIP every Blend pipeline outright
  // (ForwardIntegrationPass.cpp:232), so a translucent geom would simply not be
  // drawn. It is derived from the scene and the mjVIS_TRANSPARENT flag, never
  // from anything that moves per frame (see the plan's R7).
  bool built_valid = false;
  bourbon::DrawSubmission::Mode built_mode = bourbon::DrawSubmission::Mode::Direct;
  uint64_t built_shadow_morphology = 0;
  bourbon::OITKind built_oit = bourbon::OITKind::None;
  bourbon::OITKind oit = bourbon::OITKind::None;

  // Debug draw mode. bourbon's forward path chooses the real integration task
  // vs a debug-viz surface by the surface Token at construction
  // (ForwardIntegrator.cpp:41: surface=="lit" integrates, anything else builds a
  // ForwardVizTask), so lit-vs-viz is structural and lives in the pass-graph
  // key. `viz_depth` (mjRND_DEPTH) renders bourbon's "depth" surface -- the one
  // debug surface that maps to a MuJoCo render flag. Wireframe and segmentation
  // have no bourbon path (no exposed fill mode; the id-colour surface needs a
  // separate IBuffer integrator chain), so they are greyed out in the GUI
  // rather than wired here. The depth surface writes window-space z in [0,1]
  // into the HDR colour buffer, so the camera exposure is switched to Manual 1.0
  // while it is active (see ApplyExposure) -- otherwise the ~1e-4 photometric
  // exposure the ResolvePass applies would crush it to black.
  bool viz_depth = false;
  bool built_viz_depth = false;

  // Whether IndirectDraws' persistent device buffer has been produced by at
  // least one shadowless render drain. IndirectDraws produces that buffer only
  // in its "stage 1", gated on shape_kind_morphology being dirty -- which is the
  // first frame shapes appear. The shadow task's cull binds that buffer but is
  // NOT batch-ordered after IndirectDraws (its ordering is via .modifies(),
  // which is not a real graph edge), so if a shadow sweep coincides with the
  // frame stage 1 first runs, the cull reads the not-yet-produced (null) buffer
  // and asserts. So we force the first render drain after a (re)build to run
  // shadowless; the buffer it produces persists across frames, and shape
  // morphology is then clean, so subsequent shadow sweeps never re-trigger
  // stage 1. Reset on Init so a model reload re-primes.
  bool primed = false;

  // A LoadAction=Load framebuffer used to overlay ImGui on the resolved image.
  std::unique_ptr<bourbon::Framebuffer> ui_framebuffer;

  const mjModel* model = nullptr;
  Eigen::Array4f clear_color = {0.12f, 0.14f, 0.18f, 1.0f};

  // Retained scene: one entry per model geom (size ngeom; entries for
  // unsupported geom types have a null node), plus a material cache keyed on
  // the quantized glTF-PBR parameter set.
  std::vector<Renderable> geoms;
  // Whether any geom is currently bound to a Blend material. Drives the OIT
  // mode, hence the pass graph, so it is deliberately sticky within a model:
  // it goes true as soon as a translucent geom appears and only resets on Init.
  // That keeps a body fading in and out of translucency from rebuilding the
  // pass graph repeatedly.
  bool any_transparent = false;
  const bourbon::BXDFClass* pbr_bxdf = nullptr;
  const bourbon::PatternClass* pbr_pattern = nullptr;
  std::map<MaterialSpec::Key, bourbon::RefPtr<bourbon::Material>>
      material_cache;
  // Texture maps, keyed by the staged image plus the coordinate generation
  // bound to it. The coord map is part of the TextureMap, not of the Pattern,
  // so a texture sampled with two different texgens is uploaded twice; that is
  // the price of not owning the DeviceImage ourselves (a TextureMap's image is
  // a future resolved when its TaskHeap allocates, so it cannot be handed to a
  // second TextureMap at build time).
  std::map<std::pair<ImageKey, std::array<int32_t, 12>>,
           bourbon::RefPtr<bourbon::TextureMap>>
      texture_cache;
  // TriangleMesh shape caches, so geoms sharing a mesh/hfield share one Shape.
  std::map<int, bourbon::RefPtr<bourbon::Shape>> mesh_shape_cache;
  std::map<int, bourbon::RefPtr<bourbon::Shape>> hfield_shape_cache;

  // --- Decorations. A private mjvScene is updated every frame (mjCAT_ALL),
  // extra_geoms appended as mjCAT_DECOR, and each decoration geom realized
  // through the grow-to-high-water `decor_slots` pool. `decor_scene_made` tracks
  // whether `decor_scene` has been mjv_makeScene'd for the current model.
  // `decor_shape_count` is the number of decoration ShapeInstances ever created
  // (they persist, hidden, so a nonzero count means the world holds shapes even
  // when no model geom does -- see HasShapes). `decor_emissive_nits` scales a
  // decoration's [0,1] colour to a radiance that reads at full brightness under
  // the camera's photometric exposure (decorations are unlit overlays).
  mjvScene decor_scene = {};
  bool decor_scene_made = false;
  std::vector<DecorSlot> decor_slots;
  int decor_shape_count = 0;
  float decor_emissive_nits = 0.0f;
  bourbon::RefPtr<bourbon::Shape> unit_triangle_shape;  // shared by all triangles

  // Selection highlight: every geom of the selected body has its ShapeInstance's
  // material swapped for a glow variant (its own spec plus an emissive boost).
  // `highlighted_body` is the body currently highlighted (-1 = none), so the set
  // is only reworked when the selection actually changes.
  int highlighted_body = -1;

  // Retained flex/skin surfaces, indexed by flex/skin id (sizes nflex/nskin).
  // Their world-space, deforming geometry is read each frame from the private
  // decoration mjvScene (mjv_updateScene fills flexface/flexnormal/flextexcoord
  // and skinvert/skinnormal there); see UpdateFlexSkin. A flex/skin is
  // highlighted via the same opaque emissive-boost material swap the body
  // selection uses, driven by perturb->flexselect / perturb->skinselect. Only
  // one flex or skin is highlighted at a time (body > flex > skin, matching the
  // Filament renderer), so these track the current id (-1 = none).
  std::vector<FlexSkin> flexes;  // size nflex
  std::vector<FlexSkin> skins;   // size nskin
  int highlighted_flex = -1;
  int highlighted_skin = -1;

  // Retained lights: one per mjModel light (an mjLIGHT_IMAGE slot is left
  // source-less -- its texture drives the environment light instead), then the
  // headlight, then at most one environment light. Only the first nlight
  // entries are index-aligned with mjModel.
  std::vector<Light> lights;
  // Source image of the environment light, held so it outlives the bake.
  bourbon::RefPtr<bourbon::TextureMap2D> env_source_image;

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

  void BuildPassGraph(bourbon::DrawSubmission::Mode mode) {
    // The integrator captures draws()/shadowDraws() for the current mode at
    // construction, so select it before building. shadowDraws() is non-null
    // only in a non-Direct mode.
    draw_submission->setMode(mode);

    bourbon::DeviceImage<2>::Extent extent = ToExtent(swapchain->extent());

    clear_task = bourbon::ClearFramebufferTask::Create(
        extent, kColorFormat, /*sample_count=*/1, clear_color, *render_graph,
        *queue, *renderer_context);

    bourbon::Integrator::Params integrator_params = {
        .extent = extent,
        .color_format = kColorFormat,
        .depth_format = kDepthFormat,
        .sample_count = 1,
        .oit = oit,
        .tap = bourbon::Tap::Color,
        .clustered = false,
    };
    // "lit" builds the real integration task; any other surface builds a
    // ForwardVizTask (ForwardIntegrator.cpp:41). "depth" paints window-space z.
    bourbon::ForwardVizTask::Params forward_viz = {
        .surface = viz_depth ? bourbon::Token::Get("depth")
                             : bourbon::Token::Get("lit"),
        .texture = bourbon::Token::Get("material"),
    };

    integrator = bourbon::ForwardIntegrator::Create(
        clear_task->color(), camera.get(), world.get(),
        draw_submission->draws(), draw_submission->shadowDraws(),
        bxdf_pipelines.get(), forward_viz, integrator_params, *pass_heap,
        *render_graph, *queue, *renderer_context);

    drawable_source = std::make_unique<DrawableImageSource>(*render_graph);

    resolve_task = bourbon::ResolveTask::Create(
        extent, swapchain->format(), drawable_source->output(),
        integrator->output(), *camera, *render_graph, *renderer_context);

    built_mode = mode;
    built_oit = oit;
    built_viz_depth = viz_depth;
    built_valid = true;
  }

  // Tears down and rebuilds the pass graph for a new draw-submission mode or
  // shadow morphology. The prior pass-graph GPU work may still be in flight, so
  // drain first (the trailing-fence idiom); then destroy the tasks in reverse
  // dependency order before recreating them in the same (surviving) render
  // graph. `shadow_morphology` is recorded so the caller's key check settles.
  void RebuildPassGraph(bourbon::DrawSubmission::Mode mode,
                        uint64_t shadow_morphology) {
    DrainGpu();
    resolve_task.reset();
    integrator.reset();
    drawable_source.reset();
    clear_task.reset();
    // The old tasks' pass-heap allocations are now safe to reclaim (GPU drained
    // above); free and re-commit for the rebuilt tasks.
    pass_heap->free();
    BuildPassGraph(mode);
    pass_heap->allocate();
    built_shadow_morphology = shadow_morphology;
  }

  // Sets the camera exposure for the current draw mode. The lit path uses the
  // photometric "sunny-16" preset (calibrated against the scene's physical light
  // magnitudes; see the constructor). A debug-viz surface writes plain [0,1]
  // data (e.g. depth) into the HDR buffer, which the photometric exposure (~1e-4)
  // would crush in the ResolvePass, so viz modes switch to Manual exposure 1.0;
  // ACES + sRGB in the resolve then map [0,1] to a visible ramp. All pushes are
  // change-gated, so this is a no-op on steady state.
  void ApplyExposure(bool viz) {
    if (viz) {
      camera->exposure_mode().setValueIfChanged(
          bourbon::Camera::ExposureMode::Manual);
      camera->exposure().setValueIfChanged(1.0f);
    } else {
      camera->exposure_mode().setValueIfChanged(
          bourbon::Camera::ExposureMode::Photometric);
      camera->f_number().setValueIfChanged(16.0f);
      camera->shutter_time().setValueIfChanged(1.0f / 125.0f);
      camera->iso().setValueIfChanged(100.0f);
      camera->ev_compensation().setValueIfChanged(0.0f);
    }
  }

  // Uploads a host triangle soup as a bourbon TriangleMesh Shape, deriving
  // tangents/bitangents per vertex. Returns null for an empty build.
  bourbon::RefPtr<bourbon::Shape> CreateTriangleMeshShape(const MeshBuild& mb) {
    const size_t vcount = mb.positions.size();
    if (vcount < 3) return {};
    std::vector<Eigen::Vector3f> tangents(vcount), bitangents(vcount);
    for (size_t i = 0; i < vcount; ++i) {
      ComputeTangents(mb.normals[i], tangents[i], bitangents[i]);
    }
    bourbon::TriangleMesh::Params params;
    params.count = static_cast<unsigned>(vcount / 3);
    params.positions = bourbon::MakeShapeVarSource(mb.positions.data());
    params.normals =
        bourbon::MakeShapeVar(bourbon::ShapeVarKind::Vertex, mb.normals.data());
    params.dPdU =
        bourbon::MakeShapeVar(bourbon::ShapeVarKind::Vertex, tangents.data());
    params.dPdV =
        bourbon::MakeShapeVar(bourbon::ShapeVarKind::Vertex, bitangents.data());
    if (mb.has_st && !mb.sts.empty()) {
      params.st =
          bourbon::MakeShapeVar(bourbon::ShapeVarKind::Vertex, mb.sts.data());
    }
    return bourbon::TriangleMesh::Create(params, *model_heap, *model_graph,
                                         *sg_context);
  }

  // Maps a MuJoCo geom to a bourbon Shape. Primitives become ray-marched SDFs
  // with sizes baked in from mjModel::geom_size (index 1 is the half-length for
  // capsule/cylinder; SDFs require rigid node transforms, so all sizing lives
  // here, never in the node scale). Planes, meshes, and height fields become
  // rasterized TriangleMeshes; mesh/hfield shapes are cached so instances share
  // one Shape. Returns null for geom types not yet supported.
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
      case mjGEOM_PLANE:
        return CreateTriangleMeshShape(BuildPlaneGeometry(m, i));
      case mjGEOM_MESH: {
        const int meshid = m->geom_dataid[i];
        if (meshid < 0) return {};
        auto it = mesh_shape_cache.find(meshid);
        if (it != mesh_shape_cache.end()) return it->second;
        auto shape = CreateTriangleMeshShape(BuildMeshGeometry(m, meshid));
        mesh_shape_cache[meshid] = shape;
        return shape;
      }
      case mjGEOM_HFIELD: {
        const int hid = m->geom_dataid[i];
        if (hid < 0) return {};
        auto it = hfield_shape_cache.find(hid);
        if (it != hfield_shape_cache.end()) return it->second;
        auto shape = CreateTriangleMeshShape(BuildHfieldGeometry(m, hid));
        hfield_shape_cache[hid] = shape;
        return shape;
      }
      default:
        return {};  // SDF plugin geoms, etc.: later stages.
    }
  }

  // Removes all retained scene nodes (geoms + light) from the world model.
  void ClearScene() {
    if (world) {
      world->model().destroyAllRootNodes();
    }
    geoms.clear();
    any_transparent = false;
    material_cache.clear();
    texture_cache.clear();
    mesh_shape_cache.clear();
    hfield_shape_cache.clear();
    pbr_bxdf = nullptr;
    pbr_pattern = nullptr;
    lights.clear();
    env_source_image.reset();
    highlighted_body = -1;
    // Flex/skin nodes were just destroyed by destroyAllRootNodes; drop the
    // (now dangling) bookkeeping so BuildScene rebuilds them for the new model.
    flexes.clear();
    skins.clear();
    highlighted_flex = -1;
    highlighted_skin = -1;
    // Decoration pool: the nodes were just destroyed by destroyAllRootNodes, so
    // drop the (now dangling) slot bookkeeping and free the private scene, which
    // is model-specific and remade on the next Init.
    decor_slots.clear();
    decor_shape_count = 0;
    unit_triangle_shape.reset();
    if (decor_scene_made) {
      mjv_freeScene(&decor_scene);
      decor_scene_made = false;
    }
  }

  // Builds the coord map for a 2D texture under `tg`. All of them read the
  // object-space hit position except Uv, which transforms the shape's own st.
  bourbon::RefPtr<bourbon::TextureCoordMap2D> MakeCoordMap2D(const TexGen& tg) {
    bourbon::TaskHeap& h = *model_heap;
    bourbon::TaskGraph& g = *model_graph;
    bourbon::SGContext& sg = *sg_context;
    switch (tg.kind) {
      case TexGen::Kind::Uv: {
        // TransformUVCoordMap applies the matrix to (uv, 0, 1), and to the
        // derivatives with w = 0, so a scale here keeps mip selection right.
        Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
        matrix(0, 0) = tg.scale[0];
        matrix(1, 1) = tg.scale[1];
        matrix(0, 3) = tg.offset[0];
        matrix(1, 3) = tg.offset[1];
        return bourbon::TransformUVCoordMap::Create({.matrix = matrix}, h, g,
                                                    sg);
      }
      case TexGen::Kind::Planar: {
        // st = P.xy / size + 0.5, so flip y in the pre-transform to match the
        // classic renderer's downward-running t (render_gl3.c:198-200).
        Eigen::Matrix4f flip_y = Eigen::Matrix4f::Identity();
        flip_y(1, 1) = -1.0f;
        return bourbon::PlanarCoordMap::Create(
            {.transform = flip_y,
             .space = bourbon::Space::Object,
             .size = tg.extent},
            h, g, sg);
      }
      case TexGen::Kind::Spherical:
        return bourbon::SphericalCoordMap::Create(
            {.space = bourbon::Space::Object, .radius = tg.radius}, h, g, sg);
      case TexGen::Kind::Cylindrical:
        return bourbon::CylindricalCoordMap::Create(
            {.space = bourbon::Space::Object, .height = tg.height}, h, g, sg);
    }
    return {};
  }

  // Builds the coord map for a cube texture: the object-space position, scaled
  // by the geom's half-extents when mat_texuniform asks for it.
  bourbon::RefPtr<bourbon::TextureCoordMap3D> MakeCoordMap3D(const TexGen& tg) {
    Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
    matrix(0, 0) = tg.cube_scale[0];
    matrix(1, 1) = tg.cube_scale[1];
    matrix(2, 2) = tg.cube_scale[2];
    return bourbon::TransformCoordMap3D::Create(
        {.transform = matrix, .space = bourbon::Space::Object}, *model_heap,
        *model_graph, *sg_context);
  }

  // Uploads (or reuses) the texture map for one glTFPbr slot. Returns null when
  // the slot is unbound or the source pixels turned out to be unusable.
  bourbon::RefPtr<bourbon::TextureMap> GetOrCreateTextureMap(
      const mjModel* m, const ImageKey& key, const TexGen& tg) {
    if (!key.bound()) return {};
    const auto cache_key = std::make_pair(key, tg.Key());
    auto it = texture_cache.find(cache_key);
    if (it != texture_cache.end()) return it->second;

    const TextureImage image = StageImage(m, key);
    if (image.empty()) return {};

    // Colour-carrying roles are decoded from sRGB by the sampler when the
    // texture declares that colour space; ORM and normal data is always linear.
    // This matches the classic renderer's internal-format choice
    // (render_context.c:1486).
    const int color_src = ColorSpaceSource(key);
    const bool srgb = color_src >= 0 &&
                      m->tex_colorspace[color_src] == mjCOLORSPACE_SRGB;
    const MTL::PixelFormat format = srgb ? MTL::PixelFormatRGBA8Unorm_sRGB
                                         : MTL::PixelFormatRGBA8Unorm;

    // Repeat + trilinear, as the classic renderer uploads textures
    // (render_context.c:1468-1471).
    bourbon::TextureMap::Sampler<2> sampler;
    sampler.min_filter = MTL::SamplerMinMagFilterLinear;
    sampler.mag_filter = MTL::SamplerMinMagFilterLinear;
    sampler.mip_filter = MTL::SamplerMipFilterLinear;
    sampler.wrap = {MTL::SamplerAddressModeRepeat,
                    MTL::SamplerAddressModeRepeat};
    const unsigned mipmap_count = std::max(
        1u, static_cast<unsigned>(std::log2(
                std::min(image.width, image.height))));

    bourbon::RefPtr<bourbon::TextureMap> texture_map;
    if (image.cube) {
      std::array<const void*, 6> faces{};
      for (unsigned f = 0; f < 6; ++f) {
        faces[f] = image.pixels.data() + f * image.face_bytes();
      }
      bourbon::CubeTextureMap::HostSource source;
      source.format = format;
      source.size = image.width;
      source.pixels = faces;
      source.mipmap_count = mipmap_count;
      source.generate_mipmaps = true;
      texture_map = bourbon::CubeTextureMap::Create(
          {.source = source,
           .sampler = sampler,
           .coord_map = MakeCoordMap3D(tg)},
          *model_heap, *model_graph, *sg_context);
    } else {
      bourbon::TextureMap2D::HostSource source;
      source.format = format;
      source.size = {image.width, image.height};
      source.pixels = image.pixels.data();
      source.mipmap_count = mipmap_count;
      source.generate_mipmaps = true;
      texture_map = bourbon::TextureMap2D::Create(
          {.source = source,
           .sampler = sampler,
           .coord_map = MakeCoordMap2D(tg)},
          *model_heap, *model_graph, *sg_context);
    }
    texture_cache[cache_key] = texture_map;
    return texture_map;
  }

  // Returns the glTF-PBR material for `spec`, creating and caching one per
  // distinct (quantized) parameter set so geoms that would render identically
  // share a Pattern and Material.
  //
  // Pattern::Params values are type-erased `const void*` that Pattern's
  // constructor memcpys into its argument block, so the pointed-to storage only
  // has to outlive this call.
  bourbon::RefPtr<bourbon::Material> GetOrCreateMaterial(
      const mjModel* m, const MaterialSpec& spec,
      bourbon::SemiTransparencyKind semi_transparency) {
    const MaterialSpec::Key key = spec.key(semi_transparency);
    auto it = material_cache.find(key);
    if (it != material_cache.end()) return it->second;

    // glTF's alpha_mode (0 OPAQUE, 1 MASK, 2 BLEND) drives the BXDF's opacity
    // -- the gltf_pbr graph forces opacity to 1 in OPAQUE mode, so a Blend
    // pattern whose alpha_mode stayed 0 would be fully opaque. It is a separate
    // knob from the Pattern's SemiTransparencyKind, which selects the pipeline
    // blend state; both have to agree.
    const int32_t alpha_mode =
        semi_transparency == bourbon::SemiTransparencyKind::Blend  ? 2
        : semi_transparency == bourbon::SemiTransparencyKind::Mask ? 1
                                                                   : 0;
    const float alpha_cutoff = 0.5f;
    const float transmission = 0.0f;
    const Eigen::Array4f sheen = {0.0f, 0.0f, 0.0f, 0.0f};

    using bourbon::Token;
    const bourbon::Pattern::Params params = {
        {Token::Get("baseColor"), &spec.base_color},
        {Token::Get("metallic"), &spec.metallic},
        {Token::Get("roughness"), &spec.roughness},
        {Token::Get("transmission"), &transmission},
        {Token::Get("specular"), &spec.specular},
        {Token::Get("ior"), &spec.ior},
        {Token::Get("alpha"), &spec.alpha},
        {Token::Get("alpha_mode"), &alpha_mode},
        {Token::Get("alpha_cutoff"), &alpha_cutoff},
        {Token::Get("sheen"), &sheen},
        {Token::Get("emissive"), &spec.emissive},
    };

    // Texture bindings. A slot left out of the specs stays null in the
    // pattern's TextureMaps struct, and glTFPbrPattern branches on exactly that
    // to fall back to the uniform.
    bourbon::TextureMapSet::Specs textures;
    auto bind = [&](const char* slot, const ImageKey& image) {
      if (auto map = GetOrCreateTextureMap(m, image, spec.texgen)) {
        textures.push_back({Token::Get(slot), std::move(map)});
      }
    };
    bind("baseColorMap", spec.base_color_map);
    bind("metallicRoughnessMap", spec.metallic_roughness_map);
    bind("normalMap", spec.normal_map);
    bind("emissiveMap", spec.emissive_map);

    auto material = bourbon::Material::Create(
        pbr_bxdf,
        bourbon::Pattern::Create(*pbr_pattern, params, textures,
                                 semi_transparency, *model_heap, *model_graph,
                                 *sg_context),
        *model_heap, *model_graph, *sg_context);
    material_cache[key] = material;
    return material;
  }

  // Builds the retained scene for `m`: one node/shape/material per supported
  // geom (transforms updated per frame), and the model's lights.
  void BuildScene(const mjModel* m) {
    model = m;
    if (!m) return;
    bourbon::TaskHeap& h = *model_heap;
    bourbon::TaskGraph& g = *model_graph;
    bourbon::SGContext& sg = *sg_context;
    bourbon::Model& scene = world->model();

    // glTF-PBR material templates; GetOrCreateMaterial() instantiates one per
    // distinct parameter set (see MaterialSpecForGeom).
    pbr_bxdf = renderer_context->findBXDF(bourbon::Token::Get("glTFPbrBXDF"));
    pbr_pattern =
        renderer_context->findPattern(bourbon::Token::Get("glTFPbrPattern"));

    // Decorations are unlit: their [0,1] colour is emitted as radiance. Scale it
    // to the scene's reference illuminance / pi so an overlay reads at full
    // brightness under the camera's photometric exposure (a diffuse-white
    // surface fully lit by the reference light returns ~reference/pi nits).
    decor_emissive_nits = ReferenceIlluminance(m) / kPi;

    // Report unreadable textures once for the model rather than once per geom
    // that references one; MaterialSpecForGeom then skips them silently.
    for (int t = 0; t < m->ntex; ++t) {
      if (IsEncodedTexture(m, t)) {
        mju_warning("bourbon: texture %d is KTX/encoded (%d channels, %dx%d); "
                    "only the Filament renderer decodes those, skipping it",
                    t, m->tex_nchannel[t], m->tex_width[t], m->tex_height[t]);
      }
    }

    geoms.assign(m->ngeom, Renderable{});
    for (int i = 0; i < m->ngeom; ++i) {
      bourbon::RefPtr<bourbon::Shape> shape = MakeShapeForGeom(m, i);
      if (!shape) continue;
      const MaterialSpec spec = MaterialSpecForGeom(m, i);
      // The authored alpha alone decides the initial blend behaviour;
      // mjVIS_TRANSPARENT is a per-frame flag and is folded in by
      // UpdateTransparency once the option set is known.
      const bool blended = spec.alpha < 1.0f;
      bourbon::RefPtr<bourbon::Material> material = GetOrCreateMaterial(
          m, spec,
          blended ? bourbon::SemiTransparencyKind::Blend
                  : bourbon::SemiTransparencyKind::Opaque);
      bourbon::SGNode* node = scene.createNode();
      scene.addRootNode(*node);
      auto xform = bourbon::MatrixTransformer::Create(g);
      node->setTransformer(xform);
      geoms[i].shape_instance = node->addShape(shape, material, h, g, sg);
      geoms[i].node = node;
      geoms[i].xform = xform;
      geoms[i].shape = shape;
      geoms[i].material = material;
      geoms[i].spec = spec;
      geoms[i].dynamic = IsDynamicGeom(m, i);
      geoms[i].blended = blended;
      geoms[i].applied_alpha = spec.alpha;
      if (blended) any_transparent = true;
    }

    BuildFlexSkinNodes(m);
    BuildLights(m);
  }

  // Creates one permanent node + identity MatrixTransformer per flex and skin,
  // with its base material precomputed. No Shape is attached yet: flex/skin
  // geometry is world-space and depends on the runtime option flags and the
  // current deformation, so the Shape is attached (and later swapped) per frame
  // in UpdateFlexSkin. The transform stays identity forever -- the vertices are
  // already in world space.
  void BuildFlexSkinNodes(const mjModel* m) {
    bourbon::TaskGraph& g = *model_graph;
    bourbon::Model& scene = world->model();
    auto make = [&](MaterialSpec spec) {
      FlexSkin fs;
      fs.spec = spec;
      fs.blended = spec.alpha < 1.0f;
      fs.material = GetOrCreateMaterial(
          m, spec,
          fs.blended ? bourbon::SemiTransparencyKind::Blend
                     : bourbon::SemiTransparencyKind::Opaque);
      fs.node = scene.createNode();
      scene.addRootNode(*fs.node);
      fs.xform = bourbon::MatrixTransformer::Create(g);
      fs.xform->local_matrix().setValueIfChanged(Eigen::Affine3f::Identity());
      fs.node->setTransformer(fs.xform);
      return fs;
    };
    flexes.clear();
    flexes.reserve(m->nflex);
    for (int f = 0; f < m->nflex; ++f) {
      flexes.push_back(make(MaterialSpecForFlex(m, f)));
    }
    skins.clear();
    skins.reserve(m->nskin);
    for (int s = 0; s < m->nskin; ++s) {
      skins.push_back(make(MaterialSpecForSkin(m, s)));
    }
  }

  // Whether any geom produced a renderable node. Bourbon's forward pass graph
  // dereferences a null (zero-count) shape-index buffer and asserts if the world
  // has no shapes, so callers must not evaluate the render graph in that case.
  bool HasShapes() const {
    if (decor_shape_count > 0) return true;  // pooled decor parts persist hidden
    for (const auto& r : geoms) {
      if (r.node) return true;
    }
    for (const auto& fs : flexes) {
      if (fs.has_shape) return true;
    }
    for (const auto& fs : skins) {
      if (fs.has_shape) return true;
    }
    return false;
  }

  // Pushes a light's live colour and intensity into its DGInputs, downcasting
  // by kind (the concrete light types share no polymorphic setter). Colour is
  // authored as plain RGB (no unit-luminance renormalisation) so a MuJoCo
  // diffuse of e.g. 0.8 scales the light to 80% -- bourbon multiplies colour by
  // intensity, so brightness carried in the [0,1] diffuse is preserved.
  void SetLightColorIntensity(const Light& L, const Eigen::Array3f& rgb,
                              float value,
                              bourbon::LightSource::Intensity::Unit unit) const {
    if (!L.source) return;
    bourbon::LightSource::Colour color;
    color.kind = bourbon::LightSource::Colour::Kind::RGB;
    color.rgb = rgb;
    color.normalise = false;
    const bourbon::LightSource::Intensity intensity{value, unit};
    switch (L.kind) {
      case bourbon::SGObjectKind::DistantLight: {
        auto* d = static_cast<bourbon::DistantLight*>(L.source.get());
        d->color().setValueIfChanged(color);
        d->intensity().setValueIfChanged(intensity);
      } break;
      case bourbon::SGObjectKind::SpotLight: {
        auto* s = static_cast<bourbon::SpotLight*>(L.source.get());
        s->color().setValueIfChanged(color);
        s->intensity().setValueIfChanged(intensity);
      } break;
      case bourbon::SGObjectKind::PointLight: {
        auto* p = static_cast<bourbon::PointLight*>(L.source.get());
        p->color().setValueIfChanged(color);
        p->intensity().setValueIfChanged(intensity);
      } break;
      default:
        break;
    }
  }

  // Creates and attaches one light node, wiring the shared per-frame plumbing
  // (identity transform placeholder, addLightSource, addRootNode). `source` is
  // the already-created concrete light; `kind` selects its per-frame downcast.
  void AttachLight(Light& L) {
    bourbon::Model& scene = world->model();
    L.node = scene.createNode();
    L.xform = bourbon::MatrixTransformer::Create(*model_graph);
    L.node->setTransformer(L.xform);
    L.node->addLightSource(L.source, *model_heap, *model_graph, *sg_context);
    scene.addRootNode(*L.node);
  }

  // Builds the retained lights: one bourbon light per mjModel light (mapped by
  // light_type) plus a directional headlight. Placement/orientation is driven
  // entirely by each node's transform per frame (UpdateLights); only the
  // per-kind magnitude/unit is fixed here. Replaces the old hard-coded key/fill
  // pair. Image lights (mjLIGHT_IMAGE) get a null-source slot for now -- IBL
  // lands in a later increment -- so the lights vector stays index-aligned with
  // model lights (headlight last).
  void BuildLights(const mjModel* m) {
    bourbon::TaskHeap& h = *model_heap;
    bourbon::TaskGraph& g = *model_graph;
    bourbon::SGContext& sg = *sg_context;

    // A classic scene authors no physical intensity (brightness lives in the
    // [0,1] diffuse colour); detect that to substitute the photometric fallback.
    float total_intensity = 0.0f;
    for (int i = 0; i < m->nlight; ++i) total_intensity += m->light_intensity[i];
    const bool classic = total_intensity <= 0.0f;

    // Shadow projection parameters (baked into each ShadowSource at creation --
    // they are not DGInputs). Unlike the classic renderer -- whose shadow map is
    // a 24-bit reverse-Z depth buffer biased with glPolygonOffset (self-scaling
    // depth-buffer units) -- bourbon's shadow map stores *linear normalized
    // distance* over [near, far] and applies its bias as a *fraction of
    // (far - near)* (ShadowPass.metal, ShadowSourceUtil.metal::shadow_depth). So
    // reusing the classic camera clip range (extent*[znear, zfar], a span of
    // ~50*extent) turns the default 0.005..0.05 bias into a world offset of
    // 0.25..2.5 scene-widths, shoving every contact shadow off its caster. The
    // frustum must instead bracket the scene tightly: we fit it to the scene
    // bounding sphere and derive the bias from a fixed world-space offset.
    //
    // stat.extent is the largest bounding-box side, so a sphere of radius
    // 0.5*sqrt(3)*extent (~0.866) contains the whole box.
    const float extent = std::max(static_cast<float>(m->stat.extent), 1e-3f);
    const Eigen::Vector3f scene_center(static_cast<float>(m->stat.center[0]),
                                       static_cast<float>(m->stat.center[1]),
                                       static_cast<float>(m->stat.center[2]));
    const float scene_radius = 0.866f * extent;
    // Target world-space shadow bias (constant term and grazing-angle slope).
    const float world_bias_const = 0.0015f * extent;
    const float world_bias_slope = 0.01f * extent;
    const unsigned shadow_size = static_cast<unsigned>(
        std::min<int>(m->vis.quality.shadowsize, kMaxShadowSize));
    const float shadowscale = static_cast<float>(m->vis.map.shadowscale);

    using Unit = bourbon::LightSource::Intensity::Unit;
    lights.clear();
    lights.reserve(m->nlight + 2);
    int image_light = -1;  // first mjLIGHT_IMAGE, drives the environment light
    for (int i = 0; i < m->nlight; ++i) {
      Light L;
      bourbon::LightSource::Colour color;
      color.rgb = Eigen::Array3f(m->light_diffuse[3 * i + 0],
                                 m->light_diffuse[3 * i + 1],
                                 m->light_diffuse[3 * i + 2]);

      // MuJoCo's OpenGL quadratic attenuation model (constant, linear,
      // quadratic), forwarded verbatim; bourbon's shader uses the same
      // 1/(a0 + a1 d + a2 d^2) form. Default {1,0,0} => no distance falloff.
      const Eigen::Array3f attenuation(m->light_attenuation[3 * i + 0],
                                       m->light_attenuation[3 * i + 1],
                                       m->light_attenuation[3 * i + 2]);
      const float classic_candela = kClassicPunctualCandela;

      switch (m->light_type[i]) {
        case mjLIGHT_DIRECTIONAL:
          L.kind = bourbon::SGObjectKind::DistantLight;
          L.intensity_unit = Unit::Lux;
          L.intensity_value =
              classic ? kClassicDirectionalLux : m->light_intensity[i];
          L.source = bourbon::DistantLight::Create(
              {.intensity = {L.intensity_value, L.intensity_unit},
               .color = color},
              h, g, sg);
          break;
        case mjLIGHT_SPOT: {
          L.kind = bourbon::SGObjectKind::SpotLight;
          L.intensity_unit = Unit::Candela;
          L.intensity_value = classic ? classic_candela : m->light_intensity[i];
          // MuJoCo light_cutoff is the spot half-angle in degrees; bourbon
          // coneangle is the full angle. light_softness (0..1) widens the smooth
          // falloff edge, measured inward from the outer edge.
          const float coneangle = 2.0f * m->light_cutoff[i];
          const float conedelta = m->light_softness[i] * m->light_cutoff[i];
          L.source = bourbon::SpotLight::Create(
              {.intensity = {L.intensity_value, L.intensity_unit},
               .color = color,
               .attenuation = attenuation,
               .coneangle = coneangle,
               .conedeltaangle = conedelta},
              h, g, sg);
        } break;
        case mjLIGHT_POINT:
          L.kind = bourbon::SGObjectKind::PointLight;
          L.intensity_unit = Unit::Candela;
          L.intensity_value = classic ? classic_candela : m->light_intensity[i];
          L.source = bourbon::PointLight::Create(
              {.intensity = {L.intensity_value, L.intensity_unit},
               .color = color,
               .attenuation = attenuation},
              h, g, sg);
          break;
        default:
          // mjLIGHT_IMAGE (and any future kinds) has no punctual source: keep
          // the slot so the first nlight entries stay index-aligned with
          // mjModel, and let the first image light's texture drive the single
          // environment light built below.
          if (m->light_type[i] == mjLIGHT_IMAGE && image_light < 0) {
            image_light = i;
          }
          lights.push_back(std::move(L));
          continue;
      }
      L.casts_shadow = m->light_castshadow[i] != 0;
      L.shadow_size = shadow_size;
      // Fit near/far to the scene bounding sphere along the light's view of it.
      // These are baked from the light's model-frame placement (light_pos/
      // light_dir at qpos0); a light that later moves far relative to the scene
      // would want a rebuild via the shadow structural key, but a static light
      // -- the common case -- stays tight.
      const Eigen::Vector3f lp(static_cast<float>(m->light_pos[3 * i + 0]),
                               static_cast<float>(m->light_pos[3 * i + 1]),
                               static_cast<float>(m->light_pos[3 * i + 2]));
      L.shadow_coverage = scene_radius;  // directional ortho half-extent
      L.shadow_coneangle = 2.0f * m->light_cutoff[i] * shadowscale;  // spot only
      if (m->light_type[i] == mjLIGHT_DIRECTIONAL) {
        // Ortho box centred on the light node, measured along the light axis.
        Eigen::Vector3f ld(static_cast<float>(m->light_dir[3 * i + 0]),
                           static_cast<float>(m->light_dir[3 * i + 1]),
                           static_cast<float>(m->light_dir[3 * i + 2]));
        if (ld.squaredNorm() > 1e-12f) ld.normalize();
        const float dz = (scene_center - lp).dot(ld);
        L.shadow_near = std::max(dz - scene_radius, 1e-3f * extent);
        L.shadow_far = std::max(dz + scene_radius, L.shadow_near + 1e-3f * extent);
      } else {
        // Perspective (spot) / cube (point): distance from the light position.
        const float d = (lp - scene_center).norm();
        L.shadow_near = std::max(d - scene_radius, 1e-3f * extent);
        L.shadow_far = std::max(d + scene_radius, L.shadow_near + 1e-3f * extent);
        // Widen the shadow cone to just contain the scene bounding sphere, but
        // never past the light's own illumination cone (nothing outside it is
        // lit, so nothing there needs shadowing). A blind cutoff*shadowscale is
        // too narrow for a light close to / above the model -- e.g. humanoid's
        // overhead "top" light -- so the body's extent spills outside the shadow
        // map (which reads as lit) and only the distant, low-angle spotlight
        // seems to cast, making shadows look like they come from the side.
        const float cutoff = static_cast<float>(m->light_cutoff[i]);
        const float need_half_deg =
            (scene_radius >= d) ? 90.0f
                                : std::asin(scene_radius / d) * 180.0f / kPi;
        const float half_deg =
            std::min(cutoff, std::max(cutoff * shadowscale, need_half_deg));
        L.shadow_coneangle = 2.0f * half_deg;
      }
      // Convert the target world bias to bourbon's normalized-distance units.
      const float span = L.shadow_far - L.shadow_near;
      L.shadow_bias = {world_bias_const / span, world_bias_slope / span};
      AttachLight(L);
      lights.push_back(std::move(L));
    }

    // Headlight: a directional light riding the camera, pointing along the view
    // direction (MuJoCo's headlight is directional). Its colour/enable come from
    // vis.headlight and its transform from the camera, both updated per frame.
    {
      Light L;
      L.is_headlight = true;
      L.kind = bourbon::SGObjectKind::DistantLight;
      L.intensity_unit = Unit::Lux;
      L.intensity_value = kClassicDirectionalLux;
      L.source = bourbon::DistantLight::Create(
          {.intensity = {L.intensity_value, L.intensity_unit}}, h, g, sg);
      AttachLight(L);
      lights.push_back(std::move(L));
    }

    BuildEnvironment(m, image_light);
  }

  // Builds the single image-based (environment) light: the indirect/ambient
  // term of the scene. Its source is a lat-long radiance image in nits, built
  // on the host from, in order of preference, the first mjLIGHT_IMAGE light's
  // texture, the model's skybox, or -- failing both -- a two-colour gradient
  // synthesized from MuJoCo's own ambient terms. Everything about it (the
  // radiance, and the frame that puts MuJoCo's +z at the environment's zenith)
  // is baked here: the light has no live intensity knob, and its node never
  // moves, so UpdateLights skips it.
  void BuildEnvironment(const mjModel* m, int image_light) {
    // Pick the source texture, if any.
    int texid = -1;
    if (image_light >= 0 && m->light_texid[image_light] >= 0) {
      texid = m->light_texid[image_light];
    } else {
      for (int i = 0; i < m->ntex; ++i) {
        if (m->tex_type[i] == mjTEXTURE_SKYBOX) {
          texid = i;
          break;
        }
      }
    }
    bool cube = texid >= 0 && m->tex_type[texid] != mjTEXTURE_2D;
    // Reject data this host-side sampler cannot read: KTX and other encoded
    // textures are single-channel blobs (only the Filament renderer decodes
    // them), and a cube whose faces are neither square-repeated nor a 6-face
    // strip would be sampled out of bounds. Falling through to texid < 0
    // synthesizes the gradient instead, which is strictly better than nothing.
    if (texid >= 0) {
      const int w = m->tex_width[texid], h = m->tex_height[texid];
      const bool readable = m->tex_nchannel[texid] >= 3;
      const bool sized = !cube || h == w || h == 6 * w;
      if (!readable || !sized) {
        mju_warning("bourbon: texture %d is unusable as an environment "
                    "(%d channels, %dx%d); using a synthesized gradient",
                    texid, m->tex_nchannel[texid], w, h);
        texid = -1;
        cube = false;
      }
    }

    // Radiance calibration. A textured environment's [0,1] pixels are scaled
    // to nits; an untextured one is a gradient whose upper hemisphere carries
    // MuJoCo's ambient (see kSkyboxIrradianceFraction for the derivation of
    // both). A uniform environment of radiance L delivers pi*L lux, hence the
    // division.
    const float reference = ReferenceIlluminance(m);
    float nits = 0.0f;
    Eigen::Array3f sky = Eigen::Array3f::Zero();
    Eigen::Array3f ground = Eigen::Array3f::Zero();
    if (texid >= 0) {
      // An image light may state its own illuminance; otherwise fall back to
      // the skybox calibration.
      const float lux = (image_light >= 0 && m->light_intensity[image_light] > 0)
                            ? m->light_intensity[image_light]
                            : kSkyboxIrradianceFraction * reference;
      nits = lux / kPi;
    } else {
      Eigen::Array3f ambient(m->vis.headlight.ambient[0],
                             m->vis.headlight.ambient[1],
                             m->vis.headlight.ambient[2]);
      for (int i = 0; i < m->nlight; ++i) {
        ambient += Eigen::Array3f(m->light_ambient[3 * i + 0],
                                  m->light_ambient[3 * i + 1],
                                  m->light_ambient[3 * i + 2]);
      }
      // No authored ambient: substitute a neutral fill so shadowed surfaces are
      // not pure black, matching Filament's always-on fallback environment. A
      // model that deliberately wants a black void can still get one by not
      // casting shadows; here the fill is what makes shadows read as grey.
      if (ambient.maxCoeff() <= 0.0f) {
        ambient = Eigen::Array3f::Constant(kClassicAmbientFraction);
      }
      // Tint by the haze colour, which is what MuJoCo authors reach for to
      // colour the air; it defaults to white, so this is a no-op by default.
      const Eigen::Array3f haze(m->vis.rgba.haze[0], m->vis.rgba.haze[1],
                                m->vis.rgba.haze[2]);
      sky = ambient * haze * reference / kPi;
      ground = sky * kEnvGroundFraction;
    }

    // Rasterize the lat-long source image. Pixel (x,y) is the radiance arriving
    // from the direction that MaterialX's projection maps to its centre, taken
    // back into world space through the light's own frame.
    const unsigned height = kEnvSourceHeight;
    const unsigned width = 2 * height;
    const Eigen::Matrix3f env_to_world = EnvironmentFrame().linear();
    std::vector<Eigen::Array4f> pixels(static_cast<size_t>(width) * height);
    for (unsigned y = 0; y < height; ++y) {
      for (unsigned x = 0; x < width; ++x) {
        const Eigen::Vector3f obj =
            LatLongDirection((x + 0.5f) / width, (y + 0.5f) / height);
        const Eigen::Vector3f dir = env_to_world * obj;
        Eigen::Array3f c;
        if (texid >= 0) {
          // Cube textures are sampled in object space (MuJoCo's skybox cubes
          // are y-up, which is this light's own frame); an equirectangular 2D
          // texture in world space, where up is MuJoCo's +z.
          c = nits * (cube ? SampleCubeTexture(m, texid, obj)
                           : SampleLatLongTexture(m, texid, dir));
        } else {
          // Smoothstep in the world z cosine: a soft horizon, so the gradient
          // reads as a sky dome over a ground plane rather than a hard seam.
          const float t = 0.5f * (dir.z() + 1.0f);
          const float w = t * t * (3.0f - 2.0f * t);
          c = ground + (sky - ground) * w;
        }
        pixels[static_cast<size_t>(y) * width + x] =
            Eigen::Array4f(c[0], c[1], c[2], 1.0f);
      }
    }

    bourbon::TextureMap2D::Params image_params;
    image_params.source = bourbon::TextureMap2D::HostSource{
        .format = MTL::PixelFormatRGBA32Float,
        .size = {width, height},
        .pixels = pixels.data(),
    };
    image_params.sampler.min_filter = MTL::SamplerMinMagFilterLinear;
    image_params.sampler.mag_filter = MTL::SamplerMinMagFilterLinear;
    env_source_image = bourbon::TextureMap2D::Create(
        image_params, *model_heap, *model_graph, *sg_context);

    Light L;
    L.is_environment = true;
    L.kind = bourbon::SGObjectKind::LatLongEnvironmentLight;
    L.source = bourbon::LatLongEnvironmentLight::Create(
        {.image = env_source_image,
         .irradiance_map_height = kEnvIrradianceHeight,
         .irradiance_map_sample_count = kEnvIrradianceSampleCount,
         .radiance_map_height = kEnvRadianceHeight},
        *model_heap, *model_graph, *sg_context);
    AttachLight(L);
    L.xform->local_matrix().setValue(EnvironmentFrame());
    lights.push_back(std::move(L));
  }

  // Per-frame light update: pushes each light's world placement (via its node
  // transform) and live colour/intensity. `eye`/`forward` are the camera eye
  // position and view direction, used to place the headlight.
  void UpdateLights(const mjModel* m, const mjData* d,
                    const Eigen::Vector3f& eye,
                    const Eigen::Vector3f& forward) {
    const int n = static_cast<int>(lights.size());
    for (int i = 0; i < n; ++i) {
      Light& L = lights[i];
      if (!L.source) continue;      // mjLIGHT_IMAGE slot; see BuildEnvironment
      if (L.is_environment) continue;  // static frame and radiance
      if (L.is_headlight) {
        const bool active = m->vis.headlight.active != 0;
        const Eigen::Array3f rgb(m->vis.headlight.diffuse[0],
                                 m->vis.headlight.diffuse[1],
                                 m->vis.headlight.diffuse[2]);
        L.xform->local_matrix().setValueIfChanged(LightFrame(eye, forward));
        SetLightColorIntensity(L, rgb, active ? L.intensity_value : 0.0f,
                               L.intensity_unit);
      } else {
        const bool active = m->light_active[i] != 0;
        const Eigen::Array3f rgb(m->light_diffuse[3 * i + 0],
                                 m->light_diffuse[3 * i + 1],
                                 m->light_diffuse[3 * i + 2]);
        L.xform->local_matrix().setValueIfChanged(
            LightFrame(ReadVec3(d->light_xpos + 3 * i),
                       ReadVec3(d->light_xdir + 3 * i)));
        SetLightColorIntensity(L, rgb, active ? L.intensity_value : 0.0f,
                               L.intensity_unit);
      }
    }
  }

  // Creates the ShadowSource for a light (spot/distant -> DirectionalShadow,
  // point -> CubeShadow) with its baked projection params and attaches it to the
  // light's own node, so it inherits the light's placement frame. The shadow
  // task bakes one encode chain per source at construction, so an add/remove is
  // structural (see the PassGraphKey check in RenderFrame).
  void CreateShadowForLight(Light& L) {
    bourbon::TaskHeap& h = *model_heap;
    bourbon::TaskGraph& g = *model_graph;
    bourbon::SGContext& sg = *sg_context;
    switch (L.kind) {
      case bourbon::SGObjectKind::DistantLight:
        L.shadow = bourbon::DirectionalShadow::Create(
            bourbon::DirectionalShadow::DistantParams{
                .size = L.shadow_size,
                .intensity = kShadowIntensity,
                .width = 2.0f * L.shadow_coverage,
                .height = 2.0f * L.shadow_coverage,
                .near = L.shadow_near,
                .far = L.shadow_far,
                .filter = bourbon::ShadowFilter::PCSS,
                .penumbra_size = kShadowPenumbraSize,
                .bias = L.shadow_bias},
            h, g, sg);
        break;
      case bourbon::SGObjectKind::SpotLight:
        L.shadow = bourbon::DirectionalShadow::Create(
            bourbon::DirectionalShadow::SpotParams{
                .size = L.shadow_size,
                .intensity = kShadowIntensity,
                .coneangle = L.shadow_coneangle,
                .near = L.shadow_near,
                .far = L.shadow_far,
                .filter = bourbon::ShadowFilter::PCSS,
                .penumbra_size = kShadowPenumbraSize,
                .bias = L.shadow_bias},
            h, g, sg);
        break;
      case bourbon::SGObjectKind::PointLight:
        L.shadow = bourbon::CubeShadow::Create(
            bourbon::CubeShadow::Params{.size = L.shadow_size,
                                        .intensity = kShadowIntensity,
                                        .near = L.shadow_near,
                                        .far = L.shadow_far,
                                        .bias = L.shadow_bias},
            h, g, sg);
        break;
      default:
        return;
    }
    L.shadow_instance = L.node->addShadowSource(L.shadow, h, g, sg);
  }

  // Rebinds any geom whose effective alpha no longer matches the material it
  // holds, and reports whether it rebound anything.
  //
  // A Pattern's SemiTransparencyKind is a construction argument with no setter,
  // and a ShapeInstance's material is bound by addShape with no setter either,
  // so becoming translucent means building a second material and swapping the
  // instance -- never mutating the Pattern. GetOrCreateMaterial keys on the
  // kind, so the opaque and blend variants coexist in the cache and a geom that
  // fades in and out only pays for the swap.
  //
  // `transparent` is mjVIS_TRANSPARENT, which fades dynamic-category geoms by
  // vis.map.alpha (setMaterial, engine_vis_visualize.c:251). It is a
  // user-toggled flag, so the alpha it produces is stable frame to frame; the
  // comparisons below are what keep this from touching the scene graph every
  // frame.
  // Syncs each geom node's visibility to the mjvOption geom-group / mjVIS_STATIC
  // filters. A hidden node is dropped from the draw but stays a world shape (as
  // hidden decorations do), so this never trips the zero-shape guard. Toggling
  // node visibility does not add/remove shapes, so it needs no shadow re-prime.
  // A null option leaves everything visible (the prior always-drawn behaviour).
  void UpdateVisibility(const mjModel* m, const mjvOption* opt) {
    for (int i = 0; i < static_cast<int>(geoms.size()); ++i) {
      Renderable& r = geoms[i];
      if (!r.node) continue;
      const bool vis = opt ? GeomVisibleUnderOption(m, *opt, i) : true;
      if (vis != r.visible) {
        r.node->setVisibility(vis);
        r.visible = vis;
      }
    }
  }

  bool UpdateTransparency(const mjModel* m, bool transparent) {
    const float fade = static_cast<float>(m->vis.map.alpha);
    bool swapped = false;
    for (int i = 0; i < static_cast<int>(geoms.size()); ++i) {
      Renderable& r = geoms[i];
      if (!r.node) continue;
      // A highlighted geom's instance holds the glow material; leave it. Its
      // base material stays current for restoration on deselect (the common case
      // has no alpha change while a body is held selected).
      if (r.highlighted) continue;
      float alpha = r.spec.base_color[3];
      if (transparent && r.dynamic) alpha *= fade;
      const bool blend = alpha < 1.0f;
      if (blend == r.blended && alpha == r.applied_alpha) continue;

      MaterialSpec spec = r.spec;
      spec.base_color[3] = alpha;
      spec.alpha = alpha;
      auto material = GetOrCreateMaterial(
          m, spec,
          blend ? bourbon::SemiTransparencyKind::Blend
                : bourbon::SemiTransparencyKind::Opaque);

      r.node->removeSGObject(r.shape_instance);
      r.shape_instance =
          r.node->addShape(r.shape, material, *model_heap, *model_graph,
                           *sg_context);
      r.material = material;
      r.blended = blend;
      r.applied_alpha = alpha;
      if (blend) any_transparent = true;
      swapped = true;
    }
    return swapped;
  }

  // Adds/removes shadow sources to match the global shadow enable (mjRND_SHADOW)
  // for every shadow-casting model light (the headlight never casts). Mutates
  // the model graph structure (shadow_source_morphology), so it must run before
  // the model graph is evaluated. Returns true if any shadow source is live.
  bool UpdateShadows(bool enabled) {
    bool any = false;
    for (Light& L : lights) {
      if (!L.source || L.is_headlight || !L.casts_shadow) continue;
      if (enabled && !L.shadow_instance) {
        CreateShadowForLight(L);
      } else if (!enabled && L.shadow_instance) {
        L.node->removeSGObject(L.shadow_instance);
        L.shadow_instance = nullptr;
        L.shadow.reset();
      }
      any = any || (L.shadow_instance != nullptr);
    }
    return any;
  }

  // Syncs the selection highlight to `select` (a body id, or <=0 for none), the
  // value app.cc sets from a pick. Every geom of the selected body gets a second
  // ShapeInstance carrying the shared glow material; deselecting removes them.
  // Only reworked when the selection changes (adding/removing instances mutates
  // shape morphology, so a change re-primes shadows). Returns true on a change.
  // The glow variant of geom `i`'s material: its own parameter set with a warm
  // emissive boost (in nits, so it survives the exposure), at the geom's current
  // SemiTransparencyKind. Cached like any other material.
  // The glow variant of `base`: its own parameter set with a warm emissive
  // boost (in nits, so it survives the exposure), at the given blend behaviour.
  // Cached like any other material. Shared by body, flex and skin selection.
  bourbon::RefPtr<bourbon::Material> HighlightMaterial(const mjModel* m,
                                                       const MaterialSpec& base,
                                                       bool blended) {
    MaterialSpec spec = base;
    const float nits = decor_emissive_nits;
    spec.emissive[0] += kHighlightColor[0] * nits;
    spec.emissive[1] += kHighlightColor[1] * nits;
    spec.emissive[2] += kHighlightColor[2] * nits;
    return GetOrCreateMaterial(m, spec,
                               blended ? bourbon::SemiTransparencyKind::Blend
                                       : bourbon::SemiTransparencyKind::Opaque);
  }

  bourbon::RefPtr<bourbon::Material> HighlightMaterialFor(const mjModel* m,
                                                          const Renderable& r) {
    return HighlightMaterial(m, r.spec, r.blended);
  }

  bool UpdateSelection(const mjModel* m, int select) {
    const int body = select > 0 ? select : -1;
    if (body == highlighted_body) return false;
    highlighted_body = body;
    bool changed = false;
    for (int i = 0; i < static_cast<int>(geoms.size()); ++i) {
      Renderable& r = geoms[i];
      if (!r.node) continue;
      const bool want = body >= 0 && m->geom_bodyid[i] == body;
      if (want && !r.highlighted) {
        auto glow = HighlightMaterialFor(m, r);
        r.node->removeSGObject(r.shape_instance);
        r.shape_instance = r.node->addShape(r.shape, glow, *model_heap,
                                            *model_graph, *sg_context);
        r.highlighted = true;
        changed = true;
      } else if (!want && r.highlighted) {
        // Restore the geom's base material (kept current by UpdateTransparency).
        r.node->removeSGObject(r.shape_instance);
        r.shape_instance = r.node->addShape(r.shape, r.material, *model_heap,
                                            *model_graph, *sg_context);
        r.highlighted = false;
        changed = true;
      }
    }
    if (changed) primed = false;
    return changed;
  }

  // --- Flex / skin -----------------------------------------------------------
  //
  // Flexes and skins are lit, retained surfaces on the opaque HDR path (like
  // model geoms, NOT the unlit decoration path). Their fully-deformed, world-
  // space geometry is already computed by mjv_updateScene into the private
  // decoration mjvScene (run in UpdateDecorations, which precedes this):
  //   - flex faces: a ready de-indexed triangle soup in decor_scene.flexface
  //     (9 floats/face), flexnormal (9/face), flextexcoord (6/face); per-flex
  //     range flexfaceadr[f], live count flexfaceused[f].
  //   - skin verts: decor_scene.skinvert (bone-blended, 3/vert) + skinnormal,
  //     de-indexed here with mjModel skin_face over skin_faceadr/facenum, and
  //     texcoords from mjModel skin_texcoord.
  // So no bone-blend / element-vs-smooth port is needed; the arrays are handed
  // straight to the shared mesh-upload path.
  //
  // The node + its identity transform are permanent (built once); only the Shape
  // is swapped, and only when the source vertices change (gated on a checksum),
  // so a static flex/skin costs just the checksum after the first build. Every
  // shape/material rebind goes through SetFlexSkinShape.

  // Removes the flex/skin's current instance and, if `shape` is non-null,
  // adopts it as the held shape; then (re)attaches the held shape with
  // `material`. A null held shape after this leaves the node shape-less (hidden
  // by absence). Any rebind re-dirties shape_kind_morphology, which re-runs
  // IndirectDraws stage 1 and reallocates the buffer the shadow cull binds, so
  // it re-primes (the same collision UpdateTransparency/UpdateSelection avoid).
  void SetFlexSkinShape(FlexSkin& fs, bourbon::RefPtr<bourbon::Shape> shape,
                        bourbon::RefPtr<bourbon::Material> material) {
    if (fs.shape_instance) {
      fs.node->removeSGObject(fs.shape_instance);
      fs.shape_instance = nullptr;
    }
    if (shape) fs.shape = shape;
    if (!fs.shape || !material) {
      fs.has_shape = false;
      primed = false;
      return;
    }
    fs.shape_instance = fs.node->addShape(fs.shape, material, *model_heap,
                                          *model_graph, *sg_context);
    fs.material = material;
    fs.has_shape = true;
    primed = false;
  }

  // Stages flex `f`'s current faces from the decoration mjvScene into `mb` and
  // sets `checksum` from the face data. Returns false when the flex has no
  // active faces (1D flex, or face/skin rendering disabled).
  bool BuildFlexMesh(int f, MeshBuild& mb, uint64_t& checksum) {
    const int nface = decor_scene.flexfaceused[f];
    if (nface <= 0) return false;
    const int adr = decor_scene.flexfaceadr[f];
    const float* face = decor_scene.flexface + 9 * adr;
    const float* norm = decor_scene.flexnormal + 9 * adr;
    const bool has_tc = model->flex_texcoordadr[f] >= 0;
    const float* tex = has_tc ? decor_scene.flextexcoord + 6 * adr : nullptr;
    checksum = HashFloats(face, static_cast<size_t>(9) * nface,
                          static_cast<uint64_t>(nface));

    const int nvert = 3 * nface;  // de-indexed triangle soup
    mb.positions.resize(nvert);
    mb.normals.resize(nvert);
    mb.has_st = has_tc;
    mb.sts.resize(has_tc ? nvert : 0);
    for (int v = 0; v < nvert; ++v) {
      mb.positions[v] = {face[3 * v], face[3 * v + 1], face[3 * v + 2]};
      mb.normals[v] = {norm[3 * v], norm[3 * v + 1], norm[3 * v + 2]};
      if (has_tc) mb.sts[v] = {tex[2 * v], tex[2 * v + 1]};
    }
    return true;
  }

  // Stages skin `s`'s current geometry: bone-blended world-space vertices and
  // smoothed normals from the decoration mjvScene, de-indexed through mjModel's
  // skin_face triangle list, with texcoords from mjModel skin_texcoord. Sets
  // `checksum` from the (deforming) vertex positions. Returns false when the
  // skin has no faces.
  bool BuildSkinMesh(int s, MeshBuild& mb, uint64_t& checksum) {
    const int facenum = model->skin_facenum[s];
    if (facenum <= 0) return false;
    const int vertadr = model->skin_vertadr[s];
    const int vertnum = model->skin_vertnum[s];
    const int faceadr = model->skin_faceadr[s];
    const float* verts = decor_scene.skinvert + 3 * vertadr;
    const float* norms = decor_scene.skinnormal + 3 * vertadr;
    const int tcadr = model->skin_texcoordadr[s];
    const bool has_tc = tcadr >= 0;
    const float* tex = has_tc ? model->skin_texcoord + 2 * tcadr : nullptr;
    checksum = HashFloats(verts, static_cast<size_t>(3) * vertnum,
                          static_cast<uint64_t>(facenum));

    const int nvert = 3 * facenum;
    mb.positions.resize(nvert);
    mb.normals.resize(nvert);
    mb.has_st = has_tc;
    mb.sts.resize(has_tc ? nvert : 0);
    for (int fi = 0; fi < facenum; ++fi) {
      const int* face = model->skin_face + 3 * (faceadr + fi);
      for (int k = 0; k < 3; ++k) {
        const int vid = face[k];  // local index within the skin
        mb.positions[3 * fi + k] = {verts[3 * vid], verts[3 * vid + 1],
                                    verts[3 * vid + 2]};
        mb.normals[3 * fi + k] = {norms[3 * vid], norms[3 * vid + 1],
                                  norms[3 * vid + 2]};
        if (has_tc) {
          mb.sts[3 * fi + k] = {tex[2 * vid], tex[2 * vid + 1]};
        }
      }
    }
    return true;
  }

  // Syncs one flex/skin surface: visibility (flags/group/alpha), geometry (Shape
  // swap gated on the checksum), and material (alpha fade + selection glow).
  // `build` stages the mesh and checksum; `visible` gates whether it draws at
  // all; `spec` is the base material spec; `highlighted` whether it carries the
  // selection glow this frame; `fade` the mjVIS_TRANSPARENT alpha multiplier.
  template <typename BuildFn>
  void UpdateOneFlexSkin(const mjModel* m, FlexSkin& fs, bool visible,
                         bool highlighted, float fade, BuildFn build) {
    if (!visible) {
      if (fs.visible) {
        fs.node->setVisibility(false);
        fs.visible = false;
      }
      return;  // keep the held Shape (hidden); no per-frame cost
    }

    MeshBuild mb;
    uint64_t checksum = 0;
    if (!build(mb, checksum)) {
      // Visible per the flags but no faces this frame: detach and hide.
      if (fs.has_shape) SetFlexSkinShape(fs, {}, {});
      if (fs.visible) {
        fs.node->setVisibility(false);
        fs.visible = false;
      }
      return;
    }

    // Effective alpha and the material it selects (fade + glow).
    float alpha = fs.spec.base_color[3] * fade;
    const bool blended = alpha < 1.0f;
    MaterialSpec spec = fs.spec;
    spec.base_color[3] = alpha;
    spec.alpha = alpha;
    const auto kind = blended ? bourbon::SemiTransparencyKind::Blend
                              : bourbon::SemiTransparencyKind::Opaque;
    bourbon::RefPtr<bourbon::Material> material =
        highlighted ? HighlightMaterial(m, spec, blended)
                    : GetOrCreateMaterial(m, spec, kind);

    const bool geom_changed = !fs.has_shape || checksum != fs.checksum;
    const bool mat_changed = material.get() != fs.material.get();
    if (geom_changed) {
      SetFlexSkinShape(fs, CreateTriangleMeshShape(mb), material);
      fs.checksum = checksum;
    } else if (mat_changed) {
      SetFlexSkinShape(fs, {}, material);  // reuse held shape, new material
    }
    fs.blended = blended;
    fs.highlighted = highlighted;
    if (blended) any_transparent = true;

    if (!fs.visible) {
      fs.node->setVisibility(true);
      fs.visible = true;
    }
  }

  // Hides every flex/skin node (keeping its held Shape). Used when the
  // decoration mjvScene could not be refreshed this frame.
  void HideAllFlexSkin() {
    for (FlexSkin& fs : flexes) {
      if (fs.visible) { fs.node->setVisibility(false); fs.visible = false; }
    }
    for (FlexSkin& fs : skins) {
      if (fs.visible) { fs.node->setVisibility(false); fs.visible = false; }
    }
  }

  // Refreshes every flex/skin surface from the decoration mjvScene (already
  // updated by UpdateDecorations this frame). Must run before the model-graph
  // evaluation, as a Shape swap mutates graph structure. Selection follows the
  // Filament priority: a selected body suppresses flex/skin highlight, then a
  // selected flex suppresses skin highlight.
  void UpdateFlexSkin(const mjModel* m, const mjvOption* opt,
                      const mjvPerturb* perturb) {
    if (!m || !decor_scene_made) return;
    mjvOption default_opt;
    if (!opt) {
      mjv_defaultOption(&default_opt);
      opt = &default_opt;
    }
    const bool transparent = opt->flags[mjVIS_TRANSPARENT] != 0;
    const float fade = transparent ? static_cast<float>(m->vis.map.alpha) : 1.0f;

    // Selection ids. A flex/skin pick sets flexselect/skinselect AND select (to
    // the flex vertex's / skin bone's body, via mjv_flexBodyId / the bone body),
    // so we must NOT gate the flex/skin glow on select being unset -- that body
    // is incidental (a flex vertex-body has no geoms, so the body glow shows
    // nothing) and gating on it would suppress the flex/skin highlight entirely.
    // A single pick sets at most one of flexselect/skinselect, so they don't
    // conflict; the body highlight (UpdateSelection, on select) runs alongside.
    const int sel_flex = perturb ? perturb->flexselect : -1;  // >=0 selects
    const int sel_skin = perturb ? perturb->skinselect : -1;  // >=0 selects
    highlighted_flex = sel_flex;
    highlighted_skin = sel_skin;

    // Flex faces exist only when face or smooth-skin rendering is enabled.
    const bool flex_faces_on =
        opt->flags[mjVIS_FLEXFACE] || opt->flags[mjVIS_FLEXSKIN];
    for (int f = 0; f < static_cast<int>(flexes.size()); ++f) {
      const int grp = m->flex_group[f];
      const int gc = grp < 0 ? 0 : (grp >= mjNGROUP ? mjNGROUP - 1 : grp);
      // A surface faded to alpha 0 is not drawn at all (classic skips it).
      const bool visible = flex_faces_on && opt->flexgroup[gc] &&
                           flexes[f].spec.base_color[3] * fade > 0.0f;
      UpdateOneFlexSkin(
          m, flexes[f], visible, f == sel_flex, fade,
          [&](MeshBuild& mb, uint64_t& cs) { return BuildFlexMesh(f, mb, cs); });
    }

    for (int s = 0; s < static_cast<int>(skins.size()); ++s) {
      const int grp = m->skin_group[s];
      const int gc = grp < 0 ? 0 : (grp >= mjNGROUP ? mjNGROUP - 1 : grp);
      const bool visible = opt->flags[mjVIS_SKIN] && opt->skingroup[gc] &&
                           skins[s].spec.base_color[3] * fade > 0.0f;
      UpdateOneFlexSkin(
          m, skins[s], visible, s == sel_skin, fade,
          [&](MeshBuild& mb, uint64_t& cs) { return BuildSkinMesh(s, mb, cs); });
    }
  }

  // --- Decoration pool -------------------------------------------------------

  // Creates the concrete bourbon shape for one decoration part, sized from the
  // layout. Sizes are baked in at construction (and re-pushed each frame by
  // ApplyDecorSize); node transforms stay rigid, so the size lives entirely in
  // the shape's own DGInputs (R2). Triangles reuse one shared unit-triangle
  // mesh; meshes reuse the model's per-dataid mesh cache.
  bourbon::RefPtr<bourbon::Shape> CreateDecorShape(const DecorPartLayout& p) {
    bourbon::TaskHeap& h = *model_heap;
    bourbon::TaskGraph& g = *model_graph;
    bourbon::SGContext& sg = *sg_context;
    switch (p.kind) {
      case DecorShapeKind::RoundCone:
        // CapsuleParams -> RoundConeSDF; a height of 0 is a sphere.
        return bourbon::SDF3D::Create(
            bourbon::SDF3D::CapsuleParams{.radius = std::max(p.radius, 1e-5f),
                                          .height = p.height},
            h, g, sg);
      case DecorShapeKind::Frustum:
        return bourbon::SDF3D::Create(
            bourbon::SDF3D::FrustumParams{.base_radius = p.base_radius,
                                          .top_radius = p.top_radius,
                                          .base_z = p.base_z,
                                          .top_z = p.top_z},
            h, g, sg);
      case DecorShapeKind::Cube:
        return bourbon::SDF3D::Create(
            bourbon::SDF3D::CubeParams{.width = std::max(p.ex, 1e-5f),
                                       .height = std::max(p.ey, 1e-5f),
                                       .depth = std::max(p.ez, 1e-5f)},
            h, g, sg);
      case DecorShapeKind::Ellipsoid:
        return bourbon::SDF3D::Create(
            bourbon::SDF3D::EllipsoidParams{.radius_x = std::max(p.ex, 1e-5f),
                                            .radius_y = std::max(p.ey, 1e-5f),
                                            .radius_z = std::max(p.ez, 1e-5f)},
            h, g, sg);
      case DecorShapeKind::BoxFrame:
        return bourbon::SDF3D::Create(
            bourbon::SDF3D::BoxFrameParams{.width = std::max(p.ex, 1e-5f),
                                           .height = std::max(p.ey, 1e-5f),
                                           .depth = std::max(p.ez, 1e-5f),
                                           .thickness = p.thickness},
            h, g, sg);
      case DecorShapeKind::Triangle: {
        if (!unit_triangle_shape) {
          MeshBuild mb;
          mb.has_st = true;
          AppendTri(mb, Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0),
                    Eigen::Vector3f(0, 1, 0), Eigen::Vector2f(0, 1),
                    Eigen::Vector2f(1, 1), Eigen::Vector2f(0, 0));
          unit_triangle_shape = CreateTriangleMeshShape(mb);
        }
        return unit_triangle_shape;
      }
      case DecorShapeKind::Mesh: {
        if (p.mesh_dataid < 0) return {};
        auto it = mesh_shape_cache.find(p.mesh_dataid);
        if (it != mesh_shape_cache.end()) return it->second;
        auto shape =
            CreateTriangleMeshShape(BuildMeshGeometry(model, p.mesh_dataid));
        mesh_shape_cache[p.mesh_dataid] = shape;
        return shape;
      }
    }
    return {};
  }

  // Pushes a decoration part's live size into its shape's DGInputs. A no-op for
  // triangles/meshes (their size is carried by the node scale / baked verts).
  void ApplyDecorSize(DecorPart& part, const DecorPartLayout& p) {
    switch (p.kind) {
      case DecorShapeKind::RoundCone: {
        auto* s = static_cast<bourbon::RoundConeSDF*>(part.shape.get());
        const float r = std::max(p.radius, 1e-5f);
        s->base_radius().setValueIfChanged(r);
        s->top_radius().setValueIfChanged(r);
        s->height().setValueIfChanged(p.height);
      } break;
      case DecorShapeKind::Frustum: {
        auto* s = static_cast<bourbon::FrustumSDF*>(part.shape.get());
        s->base_radius().setValueIfChanged(p.base_radius);
        s->top_radius().setValueIfChanged(p.top_radius);
        s->base_z().setValueIfChanged(p.base_z);
        s->top_z().setValueIfChanged(p.top_z);
      } break;
      case DecorShapeKind::Cube: {
        auto* s = static_cast<bourbon::CubeSDF*>(part.shape.get());
        s->width().setValueIfChanged(std::max(p.ex, 1e-5f));
        s->height().setValueIfChanged(std::max(p.ey, 1e-5f));
        s->depth().setValueIfChanged(std::max(p.ez, 1e-5f));
      } break;
      case DecorShapeKind::Ellipsoid: {
        auto* s = static_cast<bourbon::EllipsoidSDF*>(part.shape.get());
        s->radius_x().setValueIfChanged(std::max(p.ex, 1e-5f));
        s->radius_y().setValueIfChanged(std::max(p.ey, 1e-5f));
        s->radius_z().setValueIfChanged(std::max(p.ez, 1e-5f));
      } break;
      case DecorShapeKind::BoxFrame: {
        auto* s = static_cast<bourbon::BoxFrameSDF*>(part.shape.get());
        s->width().setValueIfChanged(std::max(p.ex, 1e-5f));
        s->height().setValueIfChanged(std::max(p.ey, 1e-5f));
        s->depth().setValueIfChanged(std::max(p.ez, 1e-5f));
        s->thickness().setValueIfChanged(p.thickness);
      } break;
      case DecorShapeKind::Triangle:
      case DecorShapeKind::Mesh:
        break;
    }
  }

  // Builds the unlit Pattern + Material for a decoration of colour `rgba`.
  // Decorations are UX overlays, so they are shaded flat: baseColor is black and
  // the colour is carried entirely by the emissive term, scaled to a radiance
  // (decor_emissive_nits) that survives the camera's photometric exposure.
  // bourbon has no unlit BXDF, so this is the glTF-PBR emissive path standing in
  // for one. A translucent decoration (alpha < 1) is a Blend material.
  void MakeDecorMaterial(DecorPart& part, const float rgba[4], bool blended) {
    const float nits = decor_emissive_nits;
    const Eigen::Array4f base_color(0.0f, 0.0f, 0.0f, rgba[3]);
    const Eigen::Array4f emissive(rgba[0] * nits, rgba[1] * nits,
                                  rgba[2] * nits, 0.0f);
    const Eigen::Array4f specular(0.0f, 0.0f, 0.0f, 0.0f);  // weight 0: unlit
    const Eigen::Array4f sheen(0.0f, 0.0f, 0.0f, 0.0f);
    const float metallic = 0.0f, roughness = 1.0f, ior = kDefaultIor;
    const float alpha = rgba[3], transmission = 0.0f, alpha_cutoff = 0.5f;
    const int32_t alpha_mode = blended ? 2 : 0;

    using bourbon::Token;
    const bourbon::Pattern::Params params = {
        {Token::Get("baseColor"), &base_color},
        {Token::Get("metallic"), &metallic},
        {Token::Get("roughness"), &roughness},
        {Token::Get("transmission"), &transmission},
        {Token::Get("specular"), &specular},
        {Token::Get("ior"), &ior},
        {Token::Get("alpha"), &alpha},
        {Token::Get("alpha_mode"), &alpha_mode},
        {Token::Get("alpha_cutoff"), &alpha_cutoff},
        {Token::Get("sheen"), &sheen},
        {Token::Get("emissive"), &emissive},
    };
    const bourbon::SemiTransparencyKind kind =
        blended ? bourbon::SemiTransparencyKind::Blend
                : bourbon::SemiTransparencyKind::Opaque;
    part.pattern = bourbon::Pattern::Create(*pbr_pattern, params, {}, kind,
                                            *model_heap, *model_graph,
                                            *sg_context);
    part.material = bourbon::Material::Create(pbr_bxdf, part.pattern,
                                              *model_heap, *model_graph,
                                              *sg_context);
    part.blended = blended;
    part.rgba_key = {-1, -1, -1, -1};  // force the first colour push
  }

  // Creates a fresh decoration part (node + rigid transform + shape + unlit
  // material). New shape kinds re-dirty shape morphology, so re-prime shadows.
  DecorPart CreateDecorPart(const DecorPartLayout& layout, const float rgba[4]) {
    DecorPart part;
    part.kind = layout.kind;
    part.shape = CreateDecorShape(layout);
    // Decorations are always opaque: bourbon's OIT/Blend path is LDR and
    // collapses to a dark smudge under the HDR photometric exposure (see the
    // note on Renderable::highlighted / MLAB), so a translucent decoration would
    // render near-black rather than semi-transparent. Opaque routes through the
    // HDR emissive path and reads correctly; a decoration's authored alpha is
    // dropped. (Flagged upstream: OIT should composite in the HDR/pre-exposure
    // space.)
    MakeDecorMaterial(part, rgba, /*blended=*/false);
    part.node = world->model().createNode();
    part.xform = bourbon::MatrixTransformer::Create(*model_graph);
    part.node->setTransformer(part.xform);
    part.instance = part.node->addShape(part.shape, part.material, *model_heap,
                                        *model_graph, *sg_context);
    world->model().addRootNode(*part.node);
    part.visible = true;
    ++decor_shape_count;
    primed = false;
    return part;
  }

  // Repurposes an existing part's node for a different shape kind: swap the
  // shape in place (the node, transform and material are kept). Mutates shape
  // morphology, so re-prime shadows.
  void RebuildDecorPartShape(DecorPart& part, const DecorPartLayout& layout) {
    part.shape = CreateDecorShape(layout);
    part.node->removeSGObject(part.instance);
    part.instance = part.node->addShape(part.shape, part.material, *model_heap,
                                        *model_graph, *sg_context);
    part.kind = layout.kind;
    primed = false;
  }

  // Pushes a decoration part's colour (and, if the alpha crossed the opacity
  // boundary, a rebuilt Blend/Opaque material). The colour goes through the
  // pattern's live DGInputs, so an unchanged colour costs nothing and a changed
  // one is a setValue rather than a graph mutation.
  void UpdateDecorColor(DecorPart& part, const float rgba[4]) {
    const std::array<int32_t, 4> key = {
        static_cast<int32_t>(std::lround(rgba[0] * 4096.0f)),
        static_cast<int32_t>(std::lround(rgba[1] * 4096.0f)),
        static_cast<int32_t>(std::lround(rgba[2] * 4096.0f)),
        static_cast<int32_t>(std::lround(rgba[3] * 4096.0f))};
    if (key == part.rgba_key) return;
    part.rgba_key = key;
    const float nits = decor_emissive_nits;
    const Eigen::Array4f emissive(rgba[0] * nits, rgba[1] * nits,
                                  rgba[2] * nits, 0.0f);
    const Eigen::Array4f base_color(0.0f, 0.0f, 0.0f, rgba[3]);
    const float alpha = rgba[3];
    using bourbon::Token;
    // NOTE: DGSpanInput now has BOTH setValue(const void*) (raw bytes, copies
    // size() bytes) and a templated setValue(const T&) (asserts size()==sizeof
    // (T)). Passing a bare `&emissive` deduces T = Array4f const* and binds the
    // template -- it then asserts sizeof(pointer)==16 and aborts. Cast to
    // const void* to select the raw-bytes overload, which is what these
    // param slices (Array4f / float) expect. (Bourbon R1 churn: the template
    // overload is newer than this call site.)
    if (auto* e = part.pattern->findParam(Token::Get("emissive"))) {
      e->setValue(static_cast<const void*>(&emissive));
    }
    if (auto* b = part.pattern->findParam(Token::Get("baseColor"))) {
      b->setValue(static_cast<const void*>(&base_color));
    }
    if (auto* a = part.pattern->findParam(Token::Get("alpha"))) {
      a->setValue(static_cast<const void*>(&alpha));
    }
  }

  void HideDecorPart(DecorPart& part) {
    if (part.visible) {
      part.node->setVisibility(false);
      part.visible = false;
    }
  }

  void HideDecorSlot(DecorSlot& slot) {
    for (DecorPart& part : slot.parts) HideDecorPart(part);
  }

  void HideAllDecorations() {
    for (DecorSlot& slot : decor_slots) HideDecorSlot(slot);
  }

  // Realizes decoration geom `geom` into pool slot `slot_index`, creating parts
  // as the high-water mark grows and hiding any surplus tail.
  void RealizeDecoration(int slot_index, const mjvGeom& geom,
                         const std::vector<DecorPartLayout>& layout) {
    if (slot_index >= static_cast<int>(decor_slots.size())) {
      decor_slots.resize(slot_index + 1);
    }
    DecorSlot& slot = decor_slots[slot_index];
    const Eigen::Affine3f world_xf = GeomAffineF(geom.pos, geom.mat);
    for (int p = 0; p < static_cast<int>(layout.size()); ++p) {
      if (p >= static_cast<int>(slot.parts.size())) {
        slot.parts.push_back(CreateDecorPart(layout[p], geom.rgba));
      }
      DecorPart& part = slot.parts[p];
      if (part.kind != layout[p].kind) RebuildDecorPartShape(part, layout[p]);
      ApplyDecorSize(part, layout[p]);
      part.xform->local_matrix().setValueIfChanged(world_xf * layout[p].local);
      UpdateDecorColor(part, geom.rgba);
      if (!part.visible) {
        part.node->setVisibility(true);
        part.visible = true;
      }
    }
    for (int p = static_cast<int>(layout.size());
         p < static_cast<int>(slot.parts.size()); ++p) {
      HideDecorPart(slot.parts[p]);
    }
  }

  // Updates every decoration from the private mjvScene. Runs mjv_updateScene
  // (mjCAT_ALL, the only way the mjvGeom API exposes decor), appends the
  // caller's extra_geoms as mjCAT_DECOR, then drives the node pool. Flex/skin
  // (mjGEOM_FLEX) is deferred to a later increment and skipped here.
  void UpdateDecorations(const mjModel* m, mjData* d,
                         const mjvPerturb* perturb, mjvCamera* cam,
                         const mjvOption* opt, const mjvGeom* extra,
                         int num_extra) {
    if (!m || !d || !cam) {
      HideAllDecorations();
      return;
    }
    if (!decor_scene_made) {
      mjv_makeScene(m, &decor_scene, kDecorSceneMaxGeom);
      decor_scene_made = true;
    }
    mjvOption default_opt;
    if (!opt) {
      mjv_defaultOption(&default_opt);
      opt = &default_opt;
    }
    mjvPerturb default_perturb;
    if (!perturb) {
      mjv_defaultPerturb(&default_perturb);
      perturb = &default_perturb;
    }
    mjv_updateScene(m, d, opt, perturb, cam, mjCAT_ALL, &decor_scene);
    const int room = decor_scene.maxgeom - decor_scene.ngeom;
    const int add = std::min(num_extra, std::max(room, 0));
    for (int i = 0; i < add; ++i) {
      decor_scene.geoms[decor_scene.ngeom] = extra[i];
      decor_scene.geoms[decor_scene.ngeom].category = mjCAT_DECOR;
      ++decor_scene.ngeom;
    }

    const float extent = std::max(static_cast<float>(m->stat.extent), 1e-3f);
    std::vector<DecorPartLayout> layout;
    int used = 0;
    for (int i = 0; i < decor_scene.ngeom; ++i) {
      const mjvGeom& geom = decor_scene.geoms[i];
      if (geom.category != mjCAT_DECOR) continue;  // flex/skin: later increment
      if (!LayoutDecoration(geom, extent, layout)) continue;
      RealizeDecoration(used++, geom, layout);
    }
    for (int s = used; s < static_cast<int>(decor_slots.size()); ++s) {
      HideDecorSlot(decor_slots[s]);
    }
  }

  // Draws geom labels into the ImGui background draw list. Labels are populated
  // on the private mjvScene's geoms by mjv_updateScene (run in UpdateDecorations)
  // whenever the option's label mode is not mjLABEL_NONE, so this reads
  // decor_scene directly. Each world position is projected with the frame's
  // clip_from_world; points behind the camera (w<=0) or outside the NDC cube are
  // skipped. Bourbon has no text path (plan R6), so this is the label renderer.
  // Must run inside the ImGui frame, before ImGui::Render().
  void DrawLabels(const mjvOption* opt) {
    if (!camera_valid || !decor_scene_made) return;
    if (!opt || opt->label == mjLABEL_NONE) return;
    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImU32 kWhite = IM_COL32(255, 255, 255, 255);
    const ImU32 kShadow = IM_COL32(0, 0, 0, 200);
    for (int i = 0; i < decor_scene.ngeom; ++i) {
      const mjvGeom& geom = decor_scene.geoms[i];
      if (geom.label[0] == '\0') continue;
      const Eigen::Vector4f clip =
          clip_from_world *
          Eigen::Vector4f(geom.pos[0], geom.pos[1], geom.pos[2], 1.0f);
      if (clip.w() <= 0.0f) continue;  // behind the camera
      const float ndc_x = clip.x() / clip.w();
      const float ndc_y = clip.y() / clip.w();
      const float ndc_z = clip.z() / clip.w();
      if (ndc_x < -1.0f || ndc_x > 1.0f || ndc_y < -1.0f || ndc_y > 1.0f ||
          ndc_z < -1.0f || ndc_z > 1.0f) {
        continue;
      }
      // NDC (y up) -> ImGui window coords (y down), centred horizontally.
      const float sx = (ndc_x * 0.5f + 0.5f) * disp.x;
      const float sy = (0.5f - ndc_y * 0.5f) * disp.y;
      const ImVec2 size = ImGui::CalcTextSize(geom.label);
      const ImVec2 pos(sx - size.x * 0.5f, sy);
      dl->AddText(ImVec2(pos.x + 1.0f, pos.y + 1.0f), kShadow, geom.label);
      dl->AddText(pos, kWhite, geom.label);
    }
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
  s.pass_heap = std::make_unique<bourbon::PersistentHeap>(*s.core);

  s.world = bourbon::World::Create(*s.model_graph, *s.render_graph,
                                   *s.renderer_context);

  // Residency sets are registered once for the lifetime of the queue.
  s.queue->addResidencySet(s.model_heap->residencySet());
  s.queue->addResidencySet(s.render_heap->residencySet());
  s.queue->addResidencySet(s.pass_heap->residencySet());
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
  s.bxdf_pipelines = bourbon::BXDFIntegrationPipelines::Create(
      *s.world, *s.render_graph, *s.renderer_context);
  s.material_pipelines = bourbon::MaterialEvaluationPipelines::Create(
      *s.world, *s.render_graph, *s.renderer_context);

  // Render in IndirectDrawsWithCull mode from the start (never Direct). Shadows
  // require a non-Direct mode (shadowDraws() is null in Direct); the WithCull
  // variant additionally frustum-culls the main geometry and is view-main's
  // intended default. (The shadow cull's dependency on IndirectDraws' device
  // buffer is handled by the priming drain -- see `primed` -- not by the draw
  // mode.)
  s.BuildPassGraph(bourbon::DrawSubmission::Mode::IndirectDrawsWithCull);
  s.pass_heap->allocate();

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
  // A reload re-dirties shape morphology, so IndirectDraws stage 1 runs again on
  // the next frame; re-prime (one shadowless drain) before re-enabling shadows.
  impl_->primed = false;
}

void BourbonContext::SetClearColor(float r, float g, float b, float a) {
  impl_->clear_color = Eigen::Array4f{r, g, b, a};
}

void BourbonContext::RenderFrame(const mjModel* model, mjData* data,
                                 const mjvPerturb* perturb, mjvCamera* mj_camera,
                                 const mjvOption* vis_option, int width,
                                 int height, const mjtByte* render_flags,
                                 const mjvGeom* extra_geoms, int num_extra_geoms) {
  Impl& s = *impl_;

  const bool shadow_enabled = render_flags && render_flags[mjRND_SHADOW];
  // Depth visualization (mjRND_DEPTH) swaps the forward path to the "depth"
  // surface; this is structural (see viz_depth) and picked up by the pass-graph
  // rebuild below. The exposure is switched to suit whichever mode is active.
  s.viz_depth = render_flags && render_flags[mjRND_DEPTH];
  s.ApplyExposure(s.viz_depth);

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
  // ImGui::Render() is deferred until after the label overlay is injected below
  // (labels need the frame's clip_from_world and the updated decoration scene).
  s.camera_valid = false;

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

    // Cache world->clip for label projection in the ImGui overlay.
    s.clip_from_world = p * world_to_eye;
    s.camera_valid = true;

    // Update the model's lights + headlight (the headlight rides the camera).
    s.UpdateLights(model, data, eye, fwd);
  }

  // --- Push per-geom world transforms into the retained scene. ---
  if (model != nullptr && data != nullptr) {
    const int n = static_cast<int>(s.geoms.size());
    for (int i = 0; i < n; ++i) {
      if (!s.geoms[i].node) continue;
      s.geoms[i].xform->local_matrix().setValueIfChanged(
          GeomAffine(data->geom_xpos + 3 * i, data->geom_xmat + 9 * i));
    }
    // Sync geom-group / mjVIS_STATIC visibility so hidden groups are not drawn
    // (matching what Pick already honours). Only touches nodes that changed.
    s.UpdateVisibility(model, vis_option);
  }

  // --- Update the pooled decorations. Runs before the HasShapes gate because a
  // scene with no model geoms can still have decorations (e.g. perturb ghosts),
  // and this may create the first decoration shapes. It mutates the model graph
  // structure, so it must precede the model-graph evaluation below.
  s.UpdateDecorations(model, data, perturb, mj_camera, vis_option, extra_geoms,
                      num_extra_geoms);

  // --- Refresh flex/skin surfaces. Reads the deformed, world-space geometry
  // mjv_updateScene just wrote into the decoration mjvScene (above), so it must
  // follow UpdateDecorations; it swaps Shapes (mutating graph structure), so it
  // must precede the model-graph evaluation below. Like decorations, it can
  // create the first shapes in a flex/skin-only scene, so it runs before the
  // HasShapes gate. Only reads the mjvScene when UpdateDecorations refreshed it
  // this frame (same preconditions); otherwise the surfaces are hidden.
  if (model != nullptr && data != nullptr && mj_camera != nullptr) {
    s.UpdateFlexSkin(model, vis_option, perturb);
  } else {
    s.HideAllFlexSkin();
  }

  // Inject geom labels into the ImGui background draw list, then finalize the
  // ImGui frame. Both must happen before the drawable is composited below.
  s.DrawLabels(vis_option);
  ImGui::Render();

  // Bourbon's forward pass graph dereferences the world's shape-index buffer
  // unconditionally, and that buffer is a null (unallocated) DevicePtr when the
  // scene has zero shapes -- evaluating the render graph then asserts inside
  // DepthAndBXDFTask. Skip the scene passes for an empty scene and just
  // composite ImGui, so an all-unsupported-geom model does not crash.
  if (s.HasShapes()) {
    // --- Sync per-geom transparency. This can rebind ShapeInstances, which
    // re-publishes shape_kind_morphology (the multiset value comes back to
    // where it was, but DGSource dirties on each assign), and a dirty shape
    // morphology re-runs IndirectDraws stage 1 -- which reallocates the buffer
    // the shadow cull binds. That is exactly the collision `primed` exists to
    // avoid, so a swap re-primes: this frame renders shadowless, the next
    // restores them.
    const bool transparent =
        vis_option != nullptr && vis_option->flags[mjVIS_TRANSPARENT] != 0;
    if (model != nullptr && s.UpdateTransparency(model, transparent)) {
      s.primed = false;
    }
    // Selection highlight: add/remove the glow overlay on the selected body.
    // Also mutates morphology (and may engage OIT), so it runs before the OIT
    // decision and the model-graph evaluation below.
    if (model != nullptr) {
      s.UpdateSelection(model, perturb != nullptr ? perturb->select : -1);
    }
    // Order-independent transparency is structural (see `built_oit`): enable it
    // as soon as the scene has, or could have, a blended geom. Both inputs are
    // stable frame to frame -- `any_transparent` is sticky per model and the
    // flag is user-toggled -- so this cannot thrash the pass graph.
    s.oit = (s.any_transparent || transparent) ? bourbon::OITKind::MLAB
                                               : bourbon::OITKind::None;

    // --- Sync shadow sources to the shadow enable. The submission mode stays
    // IndirectDrawsWithCull regardless (see the constructor): only the set of
    // shadow sources changes, which is what gates the pass-graph rebuild below.
    // Shadows are held off until the scene has been primed by one shadowless
    // render drain (see `primed`): that drain runs IndirectDraws stage 1, which
    // produces the persistent device buffer the shadow cull binds. Without this,
    // the first frame would collide stage 1 with the shadow cull in one sweep
    // and the cull would read a null buffer.
    s.UpdateShadows(shadow_enabled && s.primed);
    const bourbon::DrawSubmission::Mode desired_mode =
        bourbon::DrawSubmission::Mode::IndirectDrawsWithCull;

    // --- Evaluate the model (scene state) graph. ---
    s.model_heap->free();
    s.model_heap->allocate();
    s.model_graph_evaluator->evaluate(*s.model_heap, *s.core);

    // --- Structural pass-graph sync. The shadow-source morphology is computed
    // by the model graph just evaluated; rebuild the pass graph if the draw mode
    // or that morphology changed (an add/remove of a shadow source). setMode is
    // pushed every frame so the submission back-end matches the built graph.
    const uint64_t morphology = s.world->shadow_source_morphology().value();
    if (!s.built_valid || s.built_mode != desired_mode ||
        s.built_shadow_morphology != morphology || s.built_oit != s.oit ||
        s.built_viz_depth != s.viz_depth) {
      s.RebuildPassGraph(desired_mode, morphology);
    }
    s.draw_submission->setMode(desired_mode);

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

    // This drain ran IndirectDraws stage 1 (shapes' morphology was dirty on the
    // first frame after a build), producing the persistent device buffer. From
    // the next frame shadows may be enabled without colliding with stage 1.
    s.primed = true;
  }

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
