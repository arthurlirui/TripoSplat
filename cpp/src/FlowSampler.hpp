#pragma once

#include <cstdint>
#include <torch/torch.h>
#include <torch/script.h>

namespace triposplat::detail {

// Output of the flow-matching sampler: the denoised latent (1, 8192, 16) and
// camera (1, 1, 5) tensors (only the latent is consumed by the decoder).
struct SampledLatent {
  torch::Tensor latent;  // (1, q_token_length, in_channels)
  torch::Tensor camera;  // (1, 1, cam_channels)
};

// FlowEulerCfgSampler in C++. Mirrors triposplat.FlowEulerCfgSampler.sample:
// shift-scheduled Euler integration with classifier-free guidance.
class FlowSampler {
 public:
  FlowSampler(const std::string& weights_dir, const torch::Device& device,
              torch::ScalarType dtype);

  // cond.feature1 / cond.feature2 set the condition; neg_cond is zeros.
  // Generates the initial noise with the given seed, runs `steps` Euler steps
  // with CFG of strength `guidance_scale` and schedule `shift`.
  SampledLatent sample(const torch::Tensor& feature1,
                       const torch::Tensor& feature2,
                       uint64_t seed,
                       int steps,
                       float guidance_scale,
                       float shift);

  // Token/channel dims (mirrors FLOW_MODEL_ARGS). Used by the pipeline to size
  // the noise tensors consistently.
  static constexpr int64_t kQTokenLength = 8192;
  static constexpr int64_t kInChannels = 16;
  static constexpr int64_t kCamChannels = 5;

 private:
  torch::jit::Module flow_;
  torch::Device device_;
};

}  // namespace triposplat::detail
