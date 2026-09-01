#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "triposplat/Export.h"

namespace triposplat {

// A bundle of 3D-Gaussian attributes mirroring the Python `Gaussian` object.
// Tensors live on the pipeline's device (CUDA by default) and are contiguous.
//
// Convention (matches Python `Gaussian`):
//   - means:     (N, 3) float32, in TripoSplat's native space = raw _xyz mapped
//                 through the aabb box (xyz = _xyz*aabb_scale + aabb_min).
//                 The default world transform [[1,0,0],[0,0,-1],[0,1,0]] is
//                 applied by save_ply/save_splat at write time, NOT here.
//   - scales:    (N, 3) float32, *activated* (softplus+bias, squared+min_kernel,
//                 sqrt). Same values Python `Gaussian.get_scaling` returns.
//   - quats:     (N, 4) float32, *raw* quaternion (w, x, y, z). save_ply/save_splat
//                 normalize and transform these before writing.
//   - opacities: (N, 1) float32, *activated* (sigmoid + bias).
//   - sh0:       (N, 1, 3) float32, SH degree-0 DC coefficients.
//   - aabb:      (2, 3) float32, normalization box (min, max) applied to means.
struct Gaussian {
  torch::Tensor means;       // (N, 3)
  torch::Tensor scales;      // (N, 3) activated
  torch::Tensor quats;       // (N, 4) raw
  torch::Tensor opacities;   // (N, 1) activated
  torch::Tensor sh0;         // (N, 1, 3)
  torch::Tensor aabb;        // (2, 3)

  int64_t size() const { return means.size(0); }
};

}  // namespace triposplat
