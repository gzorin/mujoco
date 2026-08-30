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
#include <cmath>
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
#include <BourbonSG/ShapeVar.h>
#include <BourbonSG/ShapeVarKind.h>
#include <BourbonSG/Shapes/SDF3D.h>
#include <BourbonSG/Shapes/TriangleMesh.h>

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
  for (int ix = 0; ix < nx; ++ix) {
    const float x0 = -hx + 2.0f * hx * ix / nx;
    const float x1 = -hx + 2.0f * hx * (ix + 1) / nx;
    const float u0 = static_cast<float>(ix) / nx;
    const float u1 = static_cast<float>(ix + 1) / nx;
    for (int iy = 0; iy < ny; ++iy) {
      const float y0 = -hy + 2.0f * hy * iy / ny;
      const float y1 = -hy + 2.0f * hy * (iy + 1) / ny;
      const float v0 = static_cast<float>(iy) / ny;
      const float v1 = static_cast<float>(iy + 1) / ny;
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
  // TriangleMesh shape caches, so geoms sharing a mesh/hfield share one Shape.
  std::map<int, bourbon::RefPtr<bourbon::Shape>> mesh_shape_cache;
  std::map<int, bourbon::RefPtr<bourbon::Shape>> hfield_shape_cache;
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
    material_cache.clear();
    mesh_shape_cache.clear();
    hfield_shape_cache.clear();
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

  // Whether any geom produced a renderable node. Bourbon's forward pass graph
  // dereferences a null (zero-count) shape-index buffer and asserts if the world
  // has no shapes, so callers must not evaluate the render graph in that case.
  bool HasShapes() const {
    for (const auto& r : geoms) {
      if (r.node) return true;
    }
    return false;
  }

  // Adds one distant light whose illumination arrives from world direction
  // `toward_light` (the vector from a surface toward the light) at `lux`
  // illuminance. Bourbon's DistantLight stores `direction = normalize(from - to)`
  // and lights surfaces from the `to` side, so the toward-light vector is
  // `to - from`; with `from` at the origin, `to = toward_light`. (The from/to
  // naming is counterintuitive -- it is NOT the propagation direction.)
  // DistantLight is placement-independent, so an identity node transform
  // suffices.
  void AddDistantLight(const Eigen::Vector3f& toward_light, float lux) {
    bourbon::TaskHeap& h = *model_heap;
    bourbon::TaskGraph& g = *model_graph;
    bourbon::SGContext& sg = *sg_context;
    bourbon::Model& scene = world->model();

    bourbon::DistantLight::Params p;
    p.intensity = {lux, bourbon::LightSource::Intensity::Unit::Lux};
    // Colour defaults to unit-luminance white RGB.
    p.from = bourbon::Point<bourbon::Space::Object>{0.0f, 0.0f, 0.0f, 1.0f};
    p.to = bourbon::Point<bourbon::Space::Object>{toward_light.x(),
                                                  toward_light.y(),
                                                  toward_light.z(), 1.0f};
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
    // Directions are toward-light vectors (see AddDistantLight). Key: a high,
    // slightly-off-axis sun (mostly +z, MuJoCo is z-up) at clear-noon-sun
    // illuminance, paired with the camera's sunny-16 photometric exposure so the
    // scene lands in range.
    AddDistantLight(Eigen::Vector3f(-0.3f, -0.5f, 1.0f), 100000.0f);
    // Fill from the opposite side and less steep, at ~1/5 the key, to soften
    // the shadow side without flattening the form.
    AddDistantLight(Eigen::Vector3f(0.5f, 0.4f, 0.6f), 20000.0f);
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

  // Bourbon's forward pass graph dereferences the world's shape-index buffer
  // unconditionally, and that buffer is a null (unallocated) DevicePtr when the
  // scene has zero shapes -- evaluating the render graph then asserts inside
  // DepthAndBXDFTask. Skip the scene passes for an empty scene and just
  // composite ImGui, so an all-unsupported-geom model does not crash.
  if (s.HasShapes()) {
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
