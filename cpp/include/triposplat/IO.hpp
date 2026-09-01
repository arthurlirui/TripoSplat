#pragma once

#include <string>

#include "triposplat/Export.h"
#include "triposplat/Gaussian.hpp"

namespace triposplat {

// Write a standard 3DGS binary little-endian PLY. Mirrors the Python
// `Gaussian.save_ply` writer exactly:
//   vertex layout: x,y,z, nx,ny,nz, f_dc_0..2, opacity, scale_0..2, rot_0..3
//   - opacity / scale_* are stored RAW (pre-activation): opacity=logit(activated),
//     scale=log(activated). The C++ writer converts the activated values back.
//   - rot_0..3 is the normalized raw quaternion (w, x, y, z).
// Returns true on success.
TRIPOSPLAT_API bool save_ply(const std::string& path, const Gaussian& g);

// Write the 32-byte-per-splat web format (.splat). Mirrors
// `Gaussian.save_splat` / `to_splat_bytes`:
//   per splat: xyz(3xfloat32) + scale(3xfloat32) + rgba(4xuint8) + rot(4xuint8)
//   - RGB = uint8(clip(sh0*C0 + 0.5, 0, 1) * 255), A = uint8(opacity * 255)
//   - rot  = uint8(clip(quat_normalized * 128 + 128, 0, 255))
//   - splats are sorted by descending opacity * prod(scale).
// Returns true on success.
TRIPOSPLAT_API bool save_splat(const std::string& path, const Gaussian& g);

}  // namespace triposplat
