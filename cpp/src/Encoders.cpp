#include "Encoders.hpp"

#include <torch/torch.h>

#include <stdexcept>
#include <string>

namespace triposplat::detail {

namespace {

// ImageNet normalization used by DinoV3 (matches torchvision Normalize with
// mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
torch::Tensor imagenet_normalize(const torch::Tensor& img_01) {
  // img_01: (1, 3, H, W) float32 in [0, 1]
  torch::Tensor mean = torch::tensor({0.485f, 0.456f, 0.406f}, img_01.options())
                           .view({1, 3, 1, 1});
  torch::Tensor std = torch::tensor({0.229f, 0.224f, 0.225f}, img_01.options())
                          .view({1, 3, 1, 1});
  return (img_01 - mean) / std;
}

// Helper: read the dtype of a module's first parameter (the weight dtype the
// module expects for its inputs).
torch::ScalarType module_dtype(const torch::jit::Module& m) {
  for (const auto& p : m.parameters()) {
    return p.scalar_type();
  }
  return torch::kFloat;
}

torch::jit::Module load_module(const std::string& path, const torch::Device& device) {
  if (path.empty()) {
    throw std::runtime_error("Encoders: empty model path");
  }
  torch::jit::Module m = torch::jit::load(path, device);
  // Preserve each module's exported dtype (DinoV3/VAE are bfloat16, matching
  // the Python pipeline); do not force a single dtype here.
  m.eval();
  return m;
}

}  // namespace

Encoders::Encoders(const std::string& weights_dir, const torch::Device& device,
                   torch::ScalarType /*dtype*/)
    : device_(device) {
  dinov3_ = load_module(weights_dir + "/dinov3.pt", device);
  vae_ = load_module(weights_dir + "/vae_encoder.pt", device);
}

// Helper: read the dtype of a module's first parameter (the weight dtype the
// module expects for its inputs). [defined above in the anonymous namespace]

EncodedImage Encoders::encode(const torch::Tensor& img, uint64_t seed) {
  // img: (1, 3, H, W) float32 in [0, 1] on device_.
  torch::Tensor img_f32 = img.to(device_, torch::kFloat);

  // ---- DinoV3 ----
  torch::Tensor img_normed = imagenet_normalize(img_f32);
  torch::ScalarType d_dtype = module_dtype(dinov3_);
  torch::Tensor dinov3_in = img_normed.to(d_dtype);
  torch::IValue out_iv = dinov3_.forward({dinov3_in});
  torch::Tensor feature1 = out_iv.toTensor();
  // Python: F.layer_norm(dinov3_feat.float(), dinov3_feat.shape[-1:]).
  feature1 = at::layer_norm(feature1.to(torch::kFloat),
                            {feature1.size(-1)});

  // ---- Flux2 VAE ----
  // Python: vae.encode(img_tensor.to(vae_dtype) * 2 - 1, deterministic=False, generator)
  torch::ScalarType v_dtype = module_dtype(vae_);
  torch::Tensor vae_in = (img_f32.to(v_dtype) * 2.0f - 1.0f);
  torch::Tensor seed_t = torch::tensor({static_cast<int64_t>(seed)},
                                       torch::TensorOptions().dtype(torch::kInt64).device(device_));
  torch::IValue vae_iv = vae_.forward({vae_in, seed_t});
  torch::Tensor feature2 = vae_iv.toTensor();  // (1, L_vae, 128)

  // Python pads 5 zero register tokens at the front so feature2 length matches
  // feature1's (cls + 4 registers + patches).
  int64_t target_len = feature1.size(1);
  int64_t cur_len = feature2.size(1);
  int64_t pad = target_len - cur_len;
  if (pad > 0) {
    torch::Tensor zero_reg = torch::zeros(
        {feature2.size(0), pad, feature2.size(2)},
        feature2.options());
    feature2 = torch::cat({zero_reg, feature2}, /*dim=*/1);
  } else if (pad < 0) {
    // Should not happen with the standard 1024 input; truncate defensively.
    feature2 = feature2.slice(1, 0, target_len);
  }

  return EncodedImage{feature1, feature2};
}

}  // namespace triposplat::detail
