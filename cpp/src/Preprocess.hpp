#pragma once

#include <string>
#include <torch/torch.h>

namespace triposplat::detail {

// Load an image file via stb_image into a (H, W, C) uint8 CPU tensor.
// req_channels: 3 (RGB) or 4 (RGBA); 0 = keep original.
torch::Tensor load_image(const std::string& path, int req_channels);

// Decode pipeline configuration carried alongside the parsed image tensors.
struct PreprocessConfig {
  int canvas_size = 1024;
};

// Decode an input image tensor into the (1,3,H,W) float32 RGB tensor the
// encoders consume, after the Python `preprocess_image` logic:
//   - resize so min(w,h) == canvas_size (LANCZOS approximated by bilinear),
//   - if remove_background (4-channel input): erode alpha (max-pool), bbox
//     crop + center, composite on black, resize to canvas_size x canvas_size.
//   - else (3-channel input): center-pad to square, resize to canvas.
//
// Accepted input shapes (uint8 or float, HWC or CHW, with or without batch):
//   (H,W,3), (H,W,4), (1,3,H,W), (1,4,H,W). The tensor is moved to `device`.
// Returns the (1, 3, canvas_size, canvas_size) float32 tensor on `device`.
torch::Tensor preprocess_image(const torch::Tensor& image,
                               bool remove_background,
                               int erode_radius,
                               int canvas_size,
                               const torch::Device& device);

}  // namespace triposplat::detail
