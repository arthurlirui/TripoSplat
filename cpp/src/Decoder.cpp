#include "Decoder.hpp"

#include <torch/torch.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace triposplat::detail {

namespace {

constexpr int kMaxVoxelLevel = 8;

torch::jit::Module load_module(const std::string& path, const torch::Device& device) {
  if (path.empty()) throw std::runtime_error("Decoder: empty model path");
  torch::jit::Module m = torch::jit::load(path, device);
  // Preserve the exported dtype (float16 for octree/gs). The traced forwards
  // cast inputs to the module's internal dtype.
  m.eval();
  return m;
}

// Retrieve a named buffer tensor from a TorchScript module. Buffers registered
// with register_buffer are serialized as attributes accessible by name.
torch::Tensor get_buffer(const torch::jit::Module& m, const std::string& name) {
  return m.attr(name).toTensor();
}

// Mirrors triposplat.sample_probs (systematic sampling). Returns out with shape
// (*counts.shape, P): how many samples land in each of P cells per row.
torch::Tensor sample_probs(const torch::Tensor& probs_in, const torch::Tensor& counts_in) {
  // batch_shape = counts.shape (e.g. (B, prev_tokens)); flatten to rows.
  std::vector<int64_t> batch_shape = counts_in.sizes().vec();
  int64_t B = counts_in.numel();
  int64_t P = probs_in.size(-1);
  torch::Device device = probs_in.device();

  torch::Tensor probs = probs_in.view({B, P}).to(torch::kFloat32).clamp_min(0);
  torch::Tensor row_sums = probs.sum(1, /*keepdim=*/true);
  torch::Tensor zero_mask = row_sums.eq(0);
  probs = probs / row_sums.clamp_min(1);
  if (zero_mask.any().item<bool>()) {
    probs = probs.clone();
    probs.index_put_({zero_mask.expand_as(probs)}, 1.0 / static_cast<double>(P));
  }

  torch::Tensor counts = counts_in.view({B}).to(torch::kLong).to(device);
  torch::Tensor out = torch::zeros({B, P}, torch::TensorOptions().dtype(torch::kLong).device(device));
  torch::Tensor cdf = probs.cumsum(1).clamp_max(1.0 - 1e-12);

  // Process unique count values (Python batches the rand call per unique n).
  auto [unique_n, inv, unique_counts] = counts.unique_consecutive(/*return_inverse=*/true);
  (void)unique_counts;
  auto unique_acc = unique_n.accessor<int64_t, 1>();
  auto inv_acc = inv.accessor<int64_t, 1>();
  for (int64_t i = 0; i < unique_n.size(0); ++i) {
    int64_t n = unique_acc[i];
    if (n == 0) continue;
    std::vector<int64_t> rows;
    for (int64_t r = 0; r < B; ++r) {
      if (inv_acc[r] == i) rows.push_back(r);
    }
    int64_t r = static_cast<int64_t>(rows.size());
    torch::Tensor row_idx = torch::tensor(rows, torch::TensorOptions().dtype(torch::kLong).device(device));
    torch::Tensor U0 = torch::rand({r, 1}, torch::TensorOptions().device(device)) / static_cast<double>(n);
    torch::Tensor grid = torch::arange(n, torch::TensorOptions().dtype(torch::kFloat32).device(device))
                             .unsqueeze(0) / static_cast<double>(n);
    torch::Tensor us = (U0 + grid).clamp_max(1.0 - 1e-12);  // (r, n)
    torch::Tensor cdf_rows = cdf.index_select(0, row_idx);   // (r, P)
    torch::Tensor idx = torch::searchsorted(cdf_rows, us).clamp_max(P - 1);  // (r, n)
    torch::Tensor buf = torch::zeros({r, P}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    buf.scatter_add_(1, idx, torch::ones_like(idx, buf.options()));
    out.index_copy_(0, row_idx, buf.to(torch::kLong));
  }
  std::vector<int64_t> out_shape = batch_shape;
  out_shape.push_back(P);
  return out.view(out_shape);
}

// Call the traced octree module: forward(coords_norm, res, latent, num) -> logits.
torch::Tensor call_octree(torch::jit::Module& octree, const torch::Tensor& coords_norm,
                          const torch::Tensor& res, const torch::Tensor& latent,
                          const torch::Tensor& num) {
  return octree.forward({coords_norm, res, latent, num}).toTensor();
}

}  // namespace

Decoder::Decoder(const std::string& weights_dir, const torch::Device& device,
                 torch::ScalarType /*dtype*/)
    : octree_(load_module(weights_dir + "/octree.pt", device)),
      gs_(load_module(weights_dir + "/gs_decoder.pt", device)),
      device_(device) {}

torch::Tensor Decoder::octree_sample(const torch::Tensor& latent, int num_tokens) {
  int64_t B = latent.size(0);
  torch::Device device = latent.device();

  // child_offset = [[i,j,k] for k in [0,1] for j in [0,1] for i in [0,1]]
  // -> (8, 3) in order: (0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)
  torch::Tensor child_offset = torch::tensor({
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
      {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
  }, torch::TensorOptions().dtype(torch::kLong).device(device));

  torch::Tensor prev_coords_int = torch::zeros({B, 1, 3}, torch::TensorOptions().dtype(torch::kLong).device(device));
  torch::Tensor prev_counts = torch::full({B, 1}, num_tokens, torch::TensorOptions().dtype(torch::kLong).device(device));
  torch::Tensor prev_log_probs = torch::zeros({B, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::Tensor batch_range = torch::arange(B, torch::TensorOptions().dtype(torch::kLong).device(device)).unsqueeze(1);
  torch::Tensor num_tensor = torch::full({B}, num_tokens, torch::TensorOptions().dtype(torch::kLong).device(device));

  for (int lv = 1; lv <= kMaxVoxelLevel; ++lv) {
    int64_t res_p = 1LL << (lv - 1);
    int64_t res = 1LL << lv;
    torch::Tensor parent_coords_norm = (prev_coords_int.to(torch::kFloat32) + 0.5f) / static_cast<float>(res_p);
    torch::Tensor res_tensor = torch::full({B}, res, torch::TensorOptions().dtype(torch::kLong).device(device));

    torch::Tensor pred_logits = call_octree(octree_, parent_coords_norm, res_tensor, latent, num_tensor) / 1.0f;
    torch::Tensor pred_probs = torch::softmax(pred_logits, /*dim=*/-1);
    torch::Tensor pred_log_probs = torch::log_softmax(pred_logits, /*dim=*/-1);

    torch::Tensor sampled = sample_probs(pred_probs, prev_counts).flatten(1, 2);  // (B, prev*8)
    pred_log_probs = pred_log_probs.flatten(1, 2);
    torch::Tensor prev_log_probs_expanded = prev_log_probs.repeat_interleave(8, /*dim=*/1);

    // child_coords_int = (prev_coords_int[:,:,None,:] * 2 + child_offset[None,None,:,:]).flatten(1,2)
    torch::Tensor child_coords_int = (prev_coords_int.unsqueeze(2) * 2 + child_offset.view({1, 1, 8, 3})).flatten(1, 2);

    torch::Tensor mask = sampled > 0;
    int64_t max_valid = mask.sum(1).max().item<int64_t>();
    torch::Tensor scatter_indices = mask.cumsum(1) - 1;
    torch::Tensor valid_mask = mask;
    torch::Tensor valid_scatter = scatter_indices.index({valid_mask});
    torch::Tensor valid_batch = batch_range.expand_as(mask).index({valid_mask});

    torch::Tensor next_prev_coords_int = torch::zeros({B, max_valid, 3}, child_coords_int.options());
    next_prev_coords_int.index_put_({valid_batch, valid_scatter}, child_coords_int.index({mask}));

    torch::Tensor next_prev_counts = torch::zeros({B, max_valid}, sampled.options());
    next_prev_counts.index_put_({valid_batch, valid_scatter}, sampled.index({mask}));

    torch::Tensor next_prev_log_probs = torch::zeros({B, max_valid}, prev_log_probs.options());
    next_prev_log_probs.index_put_({valid_batch, valid_scatter},
                                   (prev_log_probs_expanded + pred_log_probs).index({mask}));

    prev_coords_int = next_prev_coords_int;
    prev_counts = next_prev_counts;
    prev_log_probs = next_prev_log_probs;
  }

  int64_t res = 1LL << kMaxVoxelLevel;
  // Final: repeat_interleave by counts, add uniform jitter, normalize.
  torch::Tensor coords_int = torch::repeat_interleave(prev_coords_int.flatten(0, 1),
                                                      prev_counts.flatten(0, 1), /*dim=*/0)
                                 .view({B, num_tokens, 3});
  torch::Tensor coords_norm = (coords_int.to(torch::kFloat32) +
                               torch::rand_like(coords_int.to(torch::kFloat32))) / static_cast<float>(res);
  return coords_norm;  // (B, num_tokens, 3)
}

Gaussian Decoder::decode(const torch::Tensor& latent, int num_gaussians) {
  int num_tokens = std::max(1, num_gaussians / gaussians_per_point_);

  // Sample points via the 8-level octree.
  torch::Tensor points = octree_sample(latent, num_tokens);  // (1, num_tokens, 3)

  // Run the gs decoder: forward(points, latent) -> features (1, num_tokens, out_channels).
  torch::Tensor features = gs_.forward({points, latent}).toTensor();

  return build_gaussians(points, features);
}

Gaussian Decoder::build_gaussians(const torch::Tensor& points, const torch::Tensor& features) {
  // points: (1, num_tokens, 3), features: (1, num_tokens, out_channels)
  int64_t B = points.size(0);
  int64_t num_tokens = points.size(1);
  int ng = gaussians_per_point_;
  int64_t N = B * num_tokens * ng;

  // Layout ranges (matches _calc_layout with use_learned_offset_scale && use_per_offset):
  //   _xyz         [0,   96)   (ng*3)
  //   _features_dc [96,  192)  (ng*3)
  //   _scaling     [192, 288)  (ng*3)
  //   _rotation    [288, 416)  (ng*4)
  //   _opacity     [416, 448)  (ng*1)
  //   _offset_scale[448, 480)  (ng*1)
  const int64_t r_xyz0 = 0, r_xyz1 = ng * 3;
  const int64_t r_dc0 = r_xyz1, r_dc1 = r_xyz1 + ng * 3;
  const int64_t r_sc0 = r_dc1, r_sc1 = r_dc1 + ng * 3;
  const int64_t r_rot0 = r_sc1, r_rot1 = r_sc1 + ng * 4;
  const int64_t r_op0 = r_rot1, r_op1 = r_rot1 + ng;
  const int64_t r_os0 = r_op1, r_os1 = r_op1 + ng;

  // ---- _get_offset(features) ----
  // _offset_scale = softplus(features[:,:,r_os0:r_os1].reshape(B,-1,ng,1) + base_offset_scale)
  torch::Tensor perturbation = get_buffer(gs_, "points_offset_perturbation");  // (ng, 3)
  torch::Tensor base_offset_scale = get_buffer(gs_, "base_offset_scale");       // (ng, 1) or scalar
  perturbation = perturbation.to(features.dtype());
  base_offset_scale = base_offset_scale.to(features.dtype());

  torch::Tensor offset_scale_raw = features.slice(2, r_os0, r_os1).reshape({B, num_tokens, ng, 1});
  torch::Tensor offset_scale = torch::softplus(offset_scale_raw + base_offset_scale);  // (B, nt, ng, 1)

  // offset = features[:,:,r_xyz0:r_xyz1].reshape(B, nt, ng, 3) * lr_xyz
  torch::Tensor offset = features.slice(2, r_xyz0, r_xyz1).reshape({B, num_tokens, ng, 3}) * lr_xyz_;
  if (perturb_offset_) {
    offset = offset + perturbation;  // broadcast (ng,3) over (B,nt,ng,3)
  }
  offset = torch::tanh(offset) * 0.5f * perturbe_size_;
  offset = offset * offset_scale;  // (B, nt, ng, 3)

  // ---- assemble attributes ----
  // _xyz = (offset + points[:,:,None,:]).flatten(0,1)  -> (B*nt*ng, 3) after flatten(0,1) on (B,nt,ng,3)
  torch::Tensor xyz_raw = (offset + points.unsqueeze(2));  // (B, nt, ng, 3)
  xyz_raw = xyz_raw.reshape({N, 3});

  // Apply aabb normalization: xyz = _xyz * aabb_scale + aabb_min, aabb=[-0.5,-0.5,-0.5, 1,1,1]
  // -> xyz = _xyz * 1.0 + (-0.5). Keep aabb in the features' dtype to avoid mixed-type ops.
  torch::Tensor aabb = torch::tensor({-0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f},
                                     torch::TensorOptions().dtype(features.dtype()).device(device_));
  torch::Tensor aabb_min = aabb.slice(0, 0, 3).view({1, 3});
  torch::Tensor aabb_scale = aabb.slice(0, 3, 6).view({1, 3});
  torch::Tensor means = xyz_raw * aabb_scale + aabb_min;  // (N, 3) in features dtype

  // _features_dc = features[:,:,r_dc0:r_dc1].reshape(B,nt,ng,1,3).flatten(0,1) -> (N,1,3) * lr_dc
  torch::Tensor sh0 = features.slice(2, r_dc0, r_dc1).reshape({B, num_tokens, ng, 1, 3});
  sh0 = sh0.reshape({N, 1, 3}) * lr_features_dc_;

  // _scaling (raw) = features[:,:,r_sc0:r_sc1].reshape(B,nt,ng,3).flatten(0,1) -> (N,3) * lr_scaling
  torch::Tensor scaling_raw = features.slice(2, r_sc0, r_sc1).reshape({B, num_tokens, ng, 3});
  scaling_raw = scaling_raw.reshape({N, 3}) * lr_scaling_;
  // activated: s = softplus(raw + scale_bias); s = square(s) + min_kernel^2; scaling = sqrt(s)
  // scale_bias = inverse_softplus(scaling_bias_) where inverse_softplus(x) = x + log(1 - exp(-x)).
  // min_kernel = filter_kernel_size_3d = 0.0009.
  float sb = scaling_bias_ + std::log1p(-std::exp(-scaling_bias_));  // inverse_softplus(0.004)
  torch::Tensor s_activated = torch::softplus(scaling_raw + sb);
  float mk = filter_kernel_size_3d_;
  s_activated = torch::square(s_activated) + mk * mk;
  torch::Tensor scales = torch::sqrt(s_activated);  // (N, 3) activated

  // _rotation (raw) = features[:,:,r_rot0:r_rot1].reshape(B,nt,ng,4).flatten(0,1) -> (N,4) * lr_rotation
  torch::Tensor quats = features.slice(2, r_rot0, r_rot1).reshape({B, num_tokens, ng, 4});
  quats = quats.reshape({N, 4}) * lr_rotation_;  // (N, 4) raw

  // _opacity (raw) = features[:,:,r_op0:r_op1].reshape(B,nt,ng,1).flatten(0,1) -> (N,1) * lr_opacity
  torch::Tensor opacity_raw = features.slice(2, r_op0, r_op1).reshape({B, num_tokens, ng, 1});
  opacity_raw = opacity_raw.reshape({N, 1}) * lr_opacity_;
  // activated: sigmoid(raw + opacity_bias), opacity_bias = inverse_sigmoid(0.1) = logit(0.1)
  float ob = std::log(0.1f / (1.0f - 0.1f));  // logit(0.1)
  torch::Tensor opacities = torch::sigmoid(opacity_raw + ob);  // (N, 1) activated

  // Cast to float32 for the public Gaussian struct.
  Gaussian g;
  g.means = means.to(torch::kFloat32).contiguous();
  g.scales = scales.to(torch::kFloat32).contiguous();
  g.quats = quats.to(torch::kFloat32).contiguous();
  g.opacities = opacities.to(torch::kFloat32).contiguous();
  g.sh0 = sh0.to(torch::kFloat32).contiguous();
  g.aabb = aabb.view({2, 3}).to(torch::kFloat32).contiguous();
  return g;
}

}  // namespace triposplat::detail
