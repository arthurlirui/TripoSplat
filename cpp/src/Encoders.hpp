#pragma once

#include <torch/torch.h>
#include <torch/script.h>

namespace triposplat::detail {

// Encoded image condition: feature1 (DinoV3) + feature2 (Flux2 VAE, padded with
// 5 zero register tokens to match feature1 length, as in the Python pipeline).
struct EncodedImage {
  torch::Tensor feature1;  // (1, L, 1280) float32 (DinoV3 output, layer-normed)
  torch::Tensor feature2;  // (1, L, 128)  float32 (VAE output, 5 zero regs prepended)
};

// Holds the two traced encoder modules.
class Encoders {
 public:
  Encoders(const std::string& weights_dir, const torch::Device& device,
           torch::ScalarType dtype);

  // img: (1, 3, canvas_size, canvas_size) float32 RGB in [0,1].
  // seed: passed to the VAE wrapper so the stochastic latent is reproducible.
  EncodedImage encode(const torch::Tensor& img, uint64_t seed);

 private:
  torch::jit::Module dinov3_;
  torch::jit::Module vae_;
  torch::Device device_;
};

}  // namespace triposplat::detail
