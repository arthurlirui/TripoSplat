#include "FlowSampler.hpp"

#include <torch/torch.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace triposplat::detail {

namespace {

torch::jit::Module load_module(const std::string& path, const torch::Device& device) {
  if (path.empty()) throw std::runtime_error("FlowSampler: empty model path");
  torch::jit::Module m = torch::jit::load(path, device);
  // Preserve the exported dtype (float16 for the flow model). Inputs are cast
  // to the module's dtype inside the traced forward.
  m.eval();
  return m;
}

// Call the wrapped flow model: forward(latent, camera, t, feature1, feature2)
// -> (latent_out, camera_out). Returns the two output tensors.
std::pair<torch::Tensor, torch::Tensor> call_flow(torch::jit::Module& flow,
                                                  const torch::Tensor& latent,
                                                  const torch::Tensor& camera,
                                                  const torch::Tensor& t_scaled,
                                                  const torch::Tensor& feature1,
                                                  const torch::Tensor& feature2) {
  auto out = flow.forward({latent, camera, t_scaled, feature1, feature2}).toTuple();
  return {out->elements()[0].toTensor(), out->elements()[1].toTensor()};
}

}  // namespace

FlowSampler::FlowSampler(const std::string& weights_dir, const torch::Device& device,
                         torch::ScalarType /*dtype*/)
    : flow_(load_module(weights_dir + "/flow_model.pt", device)),
      device_(device) {}

SampledLatent FlowSampler::sample(const torch::Tensor& feature1,
                                  const torch::Tensor& feature2,
                                  uint64_t seed, int steps,
                                  float guidance_scale, float shift) {
  // The C++ pipeline calls torch::manual_seed(seed) ONCE at the start of run(),
  // so the VAE's stochastic latent (drawn first) and the flow sampler's initial
  // noise (drawn here) advance the *same* global CUDA generator in sequence —
  // mirroring Python's single shared torch.Generator. We do NOT re-seed here.
  (void)seed;

  torch::Tensor latent = torch::randn({1, kQTokenLength, kInChannels},
                                      torch::TensorOptions().device(device_).dtype(torch::kFloat32));
  torch::Tensor camera = torch::randn({1, 1, kCamChannels},
                                      torch::TensorOptions().device(device_).dtype(torch::kFloat32));

  // Negative condition = zeros_like(cond).
  torch::Tensor neg_f1 = torch::zeros_like(feature1);
  torch::Tensor neg_f2 = torch::zeros_like(feature2);

  // Time schedule: t = shift * linspace(1,0,steps+1) / (1 + (shift-1)*linspace(1,0,steps+1)).
  torch::Tensor lin = torch::linspace(1.0, 0.0, steps + 1, torch::kFloat32);
  torch::Tensor t_seq = (shift * lin) / (1.0 + (shift - 1.0) * lin);
  auto t_acc = t_seq.accessor<float, 1>();

  const bool use_cfg = guidance_scale > 1.0f;

  for (int i = 0; i < steps; ++i) {
    float t = t_acc[i];
    float t_prev = t_acc[i + 1];
    float dt = t - t_prev;

    // t_scaled = tensor([1000*t]) on device (batch=1).
    torch::Tensor t_scaled = torch::full({1}, 1000.0f * t,
                                         torch::TensorOptions().device(device_).dtype(torch::kFloat32));

    // Conditional pass.
    auto [cond_lat, cond_cam] = call_flow(flow_, latent, camera, t_scaled,
                                          feature1, feature2);

    if (use_cfg) {
      auto [neg_lat, neg_cam] = call_flow(flow_, latent, camera, t_scaled,
                                          neg_f1, neg_f2);
      // pred = s * cond - (s - 1) * uncond
      float s = guidance_scale;
      latent = latent - (s * cond_lat - (s - 1.0f) * neg_lat) * dt;
      camera = camera - (s * cond_cam - (s - 1.0f) * neg_cam) * dt;
    } else {
      latent = latent - cond_lat * dt;
      camera = camera - cond_cam * dt;
    }
  }

  return SampledLatent{latent, camera};
}

}  // namespace triposplat::detail
