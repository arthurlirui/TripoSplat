#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include "triposplat/Gaussian.hpp"

namespace triposplat::detail {

// Decode the sampled latent into a Gaussian bundle. Mirrors
// `OctreeGaussianDecoder.decode` + `_build_gaussians`:
//   1. 8-level octree systematic sampling (C++ reimplementation of
//      OctreeProbabilityFixedlenDecoder.sample + sample_probs), calling the
//      traced octree module per level to get logits.
//   2. the traced gs decoder produces per-point feature vectors.
//   3. _build_gaussians assembles means/scales/quats/opacity/sh0, applies the
//      rep_config activations (softplus scaling, sigmoid opacity) and the
//      default coordinate transform [[1,0,0],[0,0,-1],[0,1,0]].
class Decoder {
 public:
  Decoder(const std::string& weights_dir, const torch::Device& device);

  // latent: (1, q_token_length, in_channels) on the pipeline device.
  // num_gaussians: target count; rounded to a multiple of gaussians_per_point.
  Gaussian decode(const torch::Tensor& latent, int num_gaussians);

  int gaussians_per_point() const { return gaussians_per_point_; }

 private:
  // 8-level octree sampling loop. Returns points_pred (1, num_tokens, 3).
  torch::Tensor octree_sample(const torch::Tensor& latent, int num_tokens);

  // Assemble Gaussian attributes from points + gs features, mirroring
  // `_build_gaussians` + the Gaussian activation setters.
  Gaussian build_gaussians(const torch::Tensor& points, const torch::Tensor& features);

  torch::jit::Module octree_;
  torch::jit::Module gs_;
  torch::Device device_;
  int gaussians_per_point_ = 32;

  // rep_config constants (mirrors GS_DECODER_ARGS.representation_config).
  float filter_kernel_size_3d_ = 0.0009f;
  float scaling_bias_ = 0.004f;
  float opacity_bias_ = 0.1f;
  float perturbe_size_ = 1.5f;
  float offset_scale_ = 0.05f;
  bool perturb_offset_ = true;
  bool use_learned_offset_scale_ = true;
  // lr multipliers per attribute.
  float lr_xyz_ = 1.0f;
  float lr_features_dc_ = 1.0f;
  float lr_opacity_ = 1.0f;
  float lr_scaling_ = 1.0f;
  float lr_rotation_ = 0.1f;
};

}  // namespace triposplat::detail
