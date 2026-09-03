<h1>
  <a href="#"><img alt="MuJoCo" src="../../../banner.png" width="100%"/></a>
</h1>

## Bourbon renderer backend (macOS)

This directory holds the **Bourbon** rendering backend for `mujoco_studio` — a
MuJoCo visualizer that draws the scene with Bourbon, a physically-based renderer
built on Metal, instead of MuJoCo's classic OpenGL renderer.

`mujoco_studio` can be built against three interchangeable backends: the classic
OpenGL renderer, Filament, and Bourbon. Bourbon is macOS-only (Metal), is
selected with `--gfx=bourbon` (`--gfx=metal` is an alias), and is the **default
backend** in any build configured with `-DMUJOCO_USE_BOURBON=ON`.

Bourbon is a *retained* renderer: the scene graph is built once at model load
and then driven per frame by pushing changed transforms into dataflow inputs, so
a body that does not move costs nothing downstream. This is the main structural
difference from the classic renderer, which rebuilds its geometry list every
frame.

### What renders today

Model geometry (SDF primitives, meshes, planes, height fields), per-geom colour
and materials, textures and texgen, transparency, model lights plus a headlight,
shadows, image-based lighting, pooled decorations, selection highlighting, geom
labels, flex and skin, and the `mjvOption` / `mjvScene` visibility and render
flags. The backend is under active development; unsupported render flags are
greyed out in the UI rather than silently ignored.

### Layout

| Path | |
|---|---|
| `bourbon_context.{h,cc}` | `BourbonContext`, a PIMPL wrapping the whole Bourbon scene graph, pass graph and Metal swapchain |
| `support/imgui_metalcpp.{h,mm}` | shim letting Dear ImGui's Metal backend talk to Bourbon's `metal-cpp` objects |
| `BOURBON_VERSION` | the Bourbon commit this backend was developed against |
| `CMakeLists.txt` | the `mujoco::render_bourbon` static library |

Every `bourbon::` call lives inside this directory. `bourbon_context.h` pulls in
no Bourbon or Metal headers, so consumers compile without Bourbon's include
directories. The rest of the integration lives outside:

| Path | |
|---|---|
| `src/experimental/platform/hal/bourbon_renderer.{h,cc}` | `platform::Renderer` implementation delegating to `BourbonContext` |
| `src/experimental/studio/` | the `mujoco_studio` application itself |
| `cmake/MujocoStudioBundle.cmake` | `.app` assembly, code signing, DMG target |
| `cmake/MujocoBundleDylib.cmake` | copies one dylib into the `.app` and normalises its `LC_RPATH`s |
| `dist/Info.plist.studio.in` | `Info.plist` template for the bundle |

---

## Requirements

- **macOS** with a Metal-capable GPU.
- **Xcode**, plus the **Metal toolchain**.
  ```sh
  xcodebuild -downloadComponent MetalToolchain
  ```
- **Bourbon**, built and installed separately. MuJoCo does not fetch it.

---

## 1. Build `mujoco_studio` (command line)

```sh
cmake -S . -B build-studio \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DMUJOCO_BUILD_STUDIO=ON \
  -DMUJOCO_USE_BOURBON=ON \
  -DMUJOCO_USE_FILAMENT=OFF \
  -DMUJOCO_STUDIO_MACOS_BUNDLE=OFF \
  -DBourbon_ROOT=/usr/local/bourbon

cmake --build build-studio --target mujoco_studio --parallel
```

`Bourbon_ROOT` is the Bourbon **install prefix**. It is the package-specific hint
`find_package(Bourbon CONFIG)` uses to locate `lib/cmake/Bourbon/BourbonConfig.cmake`,
and the same prefix `render_bourbon` resolves the bundled Eigen and the runtime
dylib rpath from. Prefer it over the broad `-DCMAKE_PREFIX_PATH`, which pollutes
the search prefix for every other package. (`-DBourbon_DIR=/usr/local/bourbon/lib/cmake/Bourbon`,
pointing directly at the config directory, also works.)

This produces a plain executable plus its assets:

```
build-studio/bin/mujoco_studio
build-studio/bin/assets/          # fonts
```

Run it:

```sh
./build-studio/bin/mujoco_studio --model_file=model/humanoid/humanoid.xml
```

Useful flags: `--gfx=` (`bourbon`/`metal`, `classic`, plus the Filament modes),
`--model_file=`, `--window_width=`, `--window_height=`. Bourbon is already the
default here, so `--gfx=bourbon` is redundant; `--gfx=classic` is the way to
compare against the classic renderer.

`MUJOCO_USE_FILAMENT=OFF` is worth keeping: Filament is a very large fetch and
build, and nothing in this backend needs it.

Note that this binary is **not relocatable**: it finds the five Bourbon dylibs
through an absolute rpath pointing at your Bourbon install. It also finds no
engine plugins, because they are built into `build-studio/lib` rather than
anywhere the launcher probes, so models needing `elasticity`, `actuator`,
`sensor` or `sdf` will fail to load. Both are fixed by the bundle build; for a
flat build, the launcher will pick the plugins up if you put them next to the
executable:

```sh
cmake --build build-studio --target elasticity actuator sensor sdf_plugin --parallel
mkdir -p build-studio/bin/mujoco_plugin
cp build-studio/lib/lib{elasticity,actuator,sensor,sdf_plugin}.dylib \
   build-studio/bin/mujoco_plugin/
```

(The `mujoco_studio` target does not build the plugins in a flat build; the
bundle build depends on them explicitly.)

---

## 2. Build the `.app` bundle

Drop `-DMUJOCO_STUDIO_MACOS_BUNDLE=OFF` (it defaults to `ON` whenever studio is
built on macOS) and build the same target:

```sh
cmake -S . -B build-studio \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DMUJOCO_BUILD_STUDIO=ON \
  -DMUJOCO_USE_BOURBON=ON \
  -DMUJOCO_USE_FILAMENT=OFF \
  -DBourbon_ROOT=/usr/local/bourbon

cmake --build build-studio --target mujoco_studio --parallel
```

The result is `build-studio/bin/MuJoCo Studio.app`, laid out like this:

```
MuJoCo Studio.app/Contents/
├── Info.plist
├── MacOS/MuJoCo Studio
├── Frameworks/                      libmujoco + the 5 Bourbon dylibs
├── PlugIns/mujoco_plugin/           elasticity, actuator, sensor, sdf
└── Resources/                       mujoco_studio.icns, assets/ (fonts)
```

Assembly happens **POST_BUILD, not at install time**, deliberately: the bundle
you launch out of the build tree during development is byte-for-byte the one
that ships, so a missing dylib cannot hide until packaging.

The bundle is self-contained apart from the Metal toolchain. Every dylib is
embedded and every load command is bundle-relative — the executable's only rpath
is `@executable_path/../Frameworks`. You can verify that:

```sh
B="build-studio/bin/MuJoCo Studio.app"
for f in "$B/Contents/MacOS/MuJoCo Studio" \
         "$B"/Contents/Frameworks/*.dylib \
         "$B"/Contents/PlugIns/mujoco_plugin/*.dylib; do
  otool -l "$f" | grep -E "^ +(path|name) /" | grep -vE "/System/|/usr/lib/"
done
```

Anything printed is a leaked absolute path and a bug. Relatedly, the absolute
rpath that plain builds use is deliberately **not** added in a bundle build: it
would let a bundle that failed to embed a dylib still resolve it from your local
Bourbon install and appear to work on your machine only.

### Code signing

Embedding rewrites load commands with `install_name_tool`, which invalidates the
ad-hoc signature the linker applied. On Apple Silicon an invalid signature is a
hard load failure, so the build re-signs the bundle inside-out automatically —
nested dylibs first, the `.app` last. Signing is not optional here.

The default identity is `-` (ad-hoc). Override it with
`-DMUJOCO_STUDIO_CODESIGN_IDENTITY="Developer ID Application: ..."`. Verify with:

```sh
codesign --verify --deep --strict --verbose=2 "build-studio/bin/MuJoCo Studio.app"
```

---

## 3. Build a DMG

```sh
cmake --build build-studio --target mujoco_studio_dmg
```

Produces `build-studio/MuJoCo-Studio-<version>.dmg`, containing the `.app`, an
`/Applications` symlink for drag-install, and a `README.txt` covering the Metal
toolchain requirement. The target is excluded from `all`.

Ad-hoc-signed builds are fine for hand-distribution but are **not notarized**,
so Gatekeeper will block them on another machine until the recipient clears
quarantine:

```sh
xattr -dr com.apple.quarantine "/Applications/MuJoCo Studio.app"
```

---

## Configure options

| Option | Default | |
|---|---|---|
| `MUJOCO_USE_BOURBON` | `OFF` | Enable this backend. macOS only; fatal error elsewhere. |
| `MUJOCO_STUDIO_MACOS_BUNDLE` | `ON` when `MUJOCO_BUILD_STUDIO` | Build `mujoco_studio` as `MuJoCo Studio.app`. Orthogonal to `MUJOCO_BUILD_MACOS_FRAMEWORKS`, which governs libmujoco's layout. |
| `MUJOCO_STUDIO_CODESIGN_IDENTITY` | `-` (ad-hoc) | Identity used to re-sign the bundle. |
| `MUJOCO_BOURBON_EIGEN_DIR` | `<Bourbon prefix>/include/eigen3` | The Eigen tree Bourbon's dylibs were compiled against. Bourbon vendors and installs its own Eigen here; must match exactly. |
| `MUJOCO_BOURBON_LLAIR_TOOLS_PATH` | from `xcrun -f metal` | Metal toolchain directory baked in as a runtime fast path. |

---

## Metal toolchain: a runtime dependency

Bourbon compiles its Metal pipelines **at runtime** through `llair`, which needs
the `metal` compiler on disk. The `.app` therefore cannot be made fully
self-contained in this one respect: a machine with only the Command Line Tools
will install it fine and then fail on first render.

`bourbon_context.cc` resolves the toolchain directory once, on first use:

1. The `MUJOCO_BOURBON_LLAIR_TOOLS` environment variable, if set. It must name a
   directory containing `metal`; if it does not, that is a hard error rather
   than a silent fallback — an explicit override that is wrong should be
   reported, not worked around.
2. The CMake-baked `MUJOCO_BOURBON_LLAIR_TOOLS_PATH`, **only if that directory
   still contains `metal`**.
3. `xcrun -f metal`.
4. Otherwise, an error naming `xcodebuild -downloadComponent MetalToolchain`.

Step 2's existence check is the load-bearing one. On recent macOS the toolchain
lives on a cryptex mount whose path carries a per-mount suffix, e.g.

```
/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.y3Nbj3/Metal.xctoolchain/usr/bin
```

That suffix is regenerated on every mount, so a path captured at configure time
goes stale across a reboot or a toolchain update — and is meaningless inside a
relocated `.app`. Checking it before use lets that case self-heal via `xcrun`
instead of hard-failing. The baked path is tried before `xcrun` only to keep a
subprocess off the common path.

---

## ABI constraints

`render_bourbon` exchanges Eigen types with Bourbon's dylibs **by value** and
relies on template symbols coalescing across the DSO boundary at load time, so
it has to match Bourbon's build ABI on four axes. `CMakeLists.txt` handles all
four; they are recorded here because the failure modes are obscure and the
fixes look arbitrary otherwise.

1. **Same Eigen tree.** A different Eigen build silently changes how fixed-size
   Eigen types are passed (e.g. `Array4f` in FP registers vs a GP register or
   memory), shifting the pointer arguments that follow and corrupting the call.
   Bourbon vendors its own Eigen and installs it into its prefix at
   `include/eigen3`; its dylibs are built against that copy, so
   `MUJOCO_BOURBON_EIGEN_DIR` defaults to it and is prepended before all other
   includes (ahead of MuJoCo's own fetched Eigen).
2. **C++20.**
3. **Default visibility.** Bourbon's DG type system identifies value types by
   the *address* of an `inline constexpr` template variable and compares those
   addresses across the boundary. MuJoCo's global hidden visibility would give
   this library its own copy at a different address, so every DG connection
   would be rejected as a type mismatch.
4. **No LTO on this target.** The visibility preset is necessary but not
   sufficient: ThinLTO's whole-program pass internalizes those same tag symbols
   at the final link, which resurrects the identical failure in Release builds
   only. `INTERPROCEDURAL_OPTIMIZATION OFF` on `render_bourbon` keeps them as
   weak-external definitions LTO cannot touch, while the rest of MuJoCo keeps
   LTO. (`-DMUJOCO_ENABLE_LTO=OFF` is an equivalent global escape hatch.)

A Debug build hides #4, so verify Release before concluding a DG type-mismatch
bug is fixed.

---

## Troubleshooting

| Symptom | |
|---|---|
| `could not locate the 'metal' tool via 'xcrun -f metal'` at configure time | Install the Metal toolchain (see [Requirements](#requirements)). |
| Error about the Metal toolchain at first render | Same, or a bad `MUJOCO_BOURBON_LLAIR_TOOLS`. |
| `find_package(Bourbon)` fails | `Bourbon_ROOT` does not point at a prefix containing `lib/cmake/Bourbon/` (or `Bourbon_DIR` not at that config directory). |
| `connect() type mismatch` at runtime | ABI mismatch — check Eigen tree and that you are testing the build you think you are (see [ABI constraints](#abi-constraints)). |
| Crash in `Create()` / allocator with garbage pointers | Eigen ABI mismatch (#1 above). |
| Compile errors in `bourbon_context.cc` after updating Bourbon | API drift; compare against the SHA in `BOURBON_VERSION`. |
| App will not launch after manual `install_name_tool` surgery | Re-sign it; an invalid signature is fatal on Apple Silicon. |
| Bundle runs on your machine but not elsewhere | A dylib is missing from `Contents/Frameworks`; run the `otool` sweep in [§3](#3-build-the-app-bundle). |
