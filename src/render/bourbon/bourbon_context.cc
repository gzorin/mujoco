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

// Derives the glTF-PBR parameters for geom `i`.
//
// This is the one place a retained renderer reading mjModel does better than
// any mjvGeom-based one: mat_metallic and mat_roughness exist in mjModel and
// are never copied into mjvGeom, so the classic and Filament paths cannot see
// them. Both default to -1 meaning "unset" (mjs_defaultMaterial), so the
// Blinn-Phong conversion remains the fallback for a geom with no material AND
// for a material that authors shininess but no roughness.
MaterialSpec MaterialSpecForGeom(const mjModel* m, int i) {
  MaterialSpec spec;
  float rgba[4];
  EffectiveGeomRgba(m, i, rgba);
  spec.base_color = {rgba[0], rgba[1], rgba[2], rgba[3]};
  spec.alpha = rgba[3];

  const int matid = m->geom_matid[i];
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
  // MuJoCo's emission is a scalar multiplier on the geom's own colour.
  spec.emissive = {emission * rgba[0], emission * rgba[1], emission * rgba[2],
                   0.0f};

  if (matid < 0) return spec;

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
    // map but the default emission of 0 would be black. Use the geom colour at
    // full strength in that case and let the map carry the variation.
    if (emission <= 0.0f) {
      spec.emissive = {rgba[0], rgba[1], rgba[2], 0.0f};
    }
  }

  // Only give the spec a coordinate generation if something actually samples
  // with it: texgen depends on the geom's own size, so letting it into the key
  // for an untextured material would fragment the cache by geom size.
  if (spec.base_color_map.bound() || spec.metallic_roughness_map.bound() ||
      spec.normal_map.bound() || spec.emissive_map.bound()) {
    spec.texgen = TexGenForGeom(m, i);
  }
  return spec;
}

// The alpha geom `i` renders with. mjVIS_TRANSPARENT fades dynamic-category
// geoms by vis.map.alpha, exactly as setMaterial does
// (engine_vis_visualize.c:251); a body is static, and so exempt, when it is
// welded to the world (bodycategory, :156).
bool IsDynamicGeom(const mjModel* m, int i) {
  return m->body_weldid[m->geom_bodyid[i]] != 0;
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

  // Transparency state. A Pattern's SemiTransparencyKind is fixed at
  // construction (Pattern only exposes getSemiTransparency()), so a geom that
  // becomes translucent needs a second, Blend-built material rather than a
  // mutated one. `spec` is the geom's parameter set at its authored (unfaded)
  // alpha; `applied_alpha` and `blended` describe the material currently bound.
  MaterialSpec spec;
  bool dynamic = false;  // mjCAT_DYNAMIC: mjVIS_TRANSPARENT fades this geom
  bool blended = false;
  float applied_alpha = 1.0f;
};

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
  // Persistent heap for the pass-graph tasks' construction-time allocations
  // (integrator, ShadowTask cull buffers, render-target rings, ...). Allocated
  // once and never freed per frame, unlike render_heap (per-frame eval scratch),
  // so those buffers are not reclaimed underneath the tasks.
  std::unique_ptr<bourbon::PersistentHeap> pass_heap;

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
    bourbon::ForwardVizTask::Params forward_viz = {
        .surface = bourbon::Token::Get("lit"),
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

    BuildLights(m);
  }

  // Whether any geom produced a renderable node. Bourbon's forward pass graph
  // dereferences a null (zero-count) shape-index buffer and asserts if the world
  // has no shapes, so callers must not evaluate the render graph in that case.
  bool HasShapes() const {
    for (const auto& r : geoms) {
      if (r.node) return true;
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
    // they are not DGInputs). Mirrors the classic/prior renderer: the shadow-map
    // near/far track the model's camera clip range, the directional ortho half-
    // extent is shadowclip*extent, and the spot shadow FOV is
    // 2*cutoff*shadowscale. Shadow-map size is clamped to a safe cap.
    const float extent = std::max(static_cast<float>(m->stat.extent), 1e-3f);
    const float shadow_near = extent * static_cast<float>(m->vis.map.znear);
    const float shadow_far = extent * static_cast<float>(m->vis.map.zfar);
    const float shadow_coverage = extent * static_cast<float>(m->vis.map.shadowclip);
    const unsigned shadow_size =
        static_cast<unsigned>(std::min(m->vis.quality.shadowsize, 4096));
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
      L.shadow_near = shadow_near;
      L.shadow_far = shadow_far;
      L.shadow_coverage = shadow_coverage;
      L.shadow_coneangle = 2.0f * m->light_cutoff[i] * shadowscale;
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
      if (ambient.maxCoeff() <= 0.0f) return;  // nothing to contribute
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
                .width = 2.0f * L.shadow_coverage,
                .height = 2.0f * L.shadow_coverage,
                .near = L.shadow_near,
                .far = L.shadow_far},
            h, g, sg);
        break;
      case bourbon::SGObjectKind::SpotLight:
        L.shadow = bourbon::DirectionalShadow::Create(
            bourbon::DirectionalShadow::SpotParams{
                .size = L.shadow_size,
                .coneangle = L.shadow_coneangle,
                .near = L.shadow_near,
                .far = L.shadow_far},
            h, g, sg);
        break;
      case bourbon::SGObjectKind::PointLight:
        L.shadow = bourbon::CubeShadow::Create(
            bourbon::CubeShadow::Params{.size = L.shadow_size,
                                        .near = L.shadow_near,
                                        .far = L.shadow_far},
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
  bool UpdateTransparency(const mjModel* m, bool transparent) {
    const float fade = static_cast<float>(m->vis.map.alpha);
    bool swapped = false;
    for (int i = 0; i < static_cast<int>(geoms.size()); ++i) {
      Renderable& r = geoms[i];
      if (!r.node) continue;
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
                                 mjvCamera* mj_camera,
                                 const mjvOption* vis_option, int width,
                                 int height, bool shadow_enabled) {
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
  }

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
        s.built_shadow_morphology != morphology || s.built_oit != s.oit) {
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
