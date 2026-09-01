#include "Preprocess.hpp"

#include <stb_image.h>

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace triposplat::detail {

namespace {

// Load an image file via stb_image into a (H, W, C) uint8 tensor on CPU.
// `req_channels` requests 3 (RGB) or 4 (RGBA); 0 = original.
torch::Tensor load_image_file(const std::string& path, int req_channels) {
  int w = 0, h = 0, c = 0;
  unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, req_channels);
  if (!data) {
    throw std::runtime_error(std::string("stb_image failed to load '") + path +
                             "': " + stbi_failure_reason());
  }
  int channels = req_channels ? req_channels : c;
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  torch::Tensor img = torch::from_blob(data, {h, w, channels}, options).clone();
  stbi_image_free(data);
  return img;
}

// Resize a (1, C, H, W) float tensor to (1, C, H_out, W_out) using bilinear
// interpolation. PIL's LANCZOS is higher quality but the downstream generative
// model is robust to the small resampling difference; bilinear keeps the
// implementation dependency-free (libtorch only).
torch::Tensor resize_bilinear(const torch::Tensor& img, int h_out, int w_out) {
  return torch::nn::functional::interpolate(
      img, torch::nn::functional::InterpolateFuncOptions({h_out, w_out})
               .mode(torch::kBilinear)
               .align_corners(false));
}

// Alpha erosion equivalent to PIL ImageFilter.MinFilter(2*r+1): a minimum
// filter on the alpha channel with a square kernel of side (2r+1). Implemented
// as max_pool2d on the *negated* alpha (min-pool == -max-pool(-x)).
torch::Tensor erode_alpha(const torch::Tensor& alpha_1hw, int radius) {
  if (radius <= 0) return alpha_1hw;
  int k = 2 * radius + 1;
  // alpha_1hw: (1, 1, H, W) float
  torch::Tensor neg = -alpha_1hw;
  torch::Tensor pooled = torch::max_pool2d(
      neg, torch::nn::functional::MaxPool2dFuncOptions(k).stride(1).padding(radius));
  return -pooled;
}

}  // namespace

torch::Tensor preprocess_image(const torch::Tensor& image_in,
                               bool remove_background,
                               int erode_radius,
                               int canvas_size,
                               const torch::Device& device) {
  // Accept a path encoded as a 1-D uint8 tensor, a file path, ... — but we only
  // support tensor inputs here. File loading is done by the caller via
  // load_image (used in Pipeline) before calling this. So `image_in` must be a
  // tensor.
  torch::Tensor image = image_in;

  // Normalize input to (H, W, C) uint8 on CPU for the resize/crop logic.
  bool is_float = image.scalar_type() == torch::kFloat ||
                  image.scalar_type() == torch::kHalf ||
                  image.scalar_type() == torch::kDouble;
  if (image.dim() == 4 && image.size(0) == 1) {
    image = image.squeeze(0);  // (C, H, W) or (H, W, C)
  }
  // Determine layout: CHW (float, 3 or 4 first) vs HWC (uint8, 3 or 4 last).
  int channels;
  torch::Tensor hwc;
  if (image.dim() == 3 && image.size(0) <= 4 && is_float) {
    // (C, H, W) -> (H, W, C)
    channels = static_cast<int>(image.size(0));
    hwc = image.permute({1, 2, 0}).contiguous();
  } else if (image.dim() == 3 && image.size(2) <= 4) {
    // (H, W, C)
    channels = static_cast<int>(image.size(2));
    hwc = image.contiguous();
  } else {
    throw std::runtime_error("preprocess_image: unsupported input shape");
  }

  if (channels != 3 && channels != 4) {
    throw std::runtime_error("preprocess_image: input must have 3 or 4 channels");
  }

  // Convert to uint8 in [0, 255] on CPU.
  if (is_float) {
    hwc = (hwc.to(torch::kFloat).clamp(0.0, 1.0) * 255.0).to(torch::kUInt8);
  } else if (image.scalar_type() != torch::kUInt8) {
    hwc = hwc.to(torch::kUInt8);
  }
  hwc = hwc.to(torch::kCPU).contiguous();

  int h = static_cast<int>(hwc.size(0));
  int w = static_cast<int>(hwc.size(1));

  // Resize so min(w, h) == canvas_size (preserve aspect). Matches Python:
  //   s = canvas / min(w, h); new = (round(w*s), round(h*s))
  float s = static_cast<float>(canvas_size) / static_cast<float>(std::min(w, h));
  int new_w = std::max(1, static_cast<int>(std::lround(w * s)));
  int new_h = std::max(1, static_cast<int>(std::lround(h * s)));

  // Move to float (1, C, H, W) for resize.
  torch::Tensor chw = hwc.permute({2, 0, 1}).to(torch::kFloat) / 255.0f;
  chw = chw.unsqueeze(0);  // (1, C, H, W)
  chw = resize_bilinear(chw, new_h, new_w);

  torch::Tensor rgb = chw.index({torch::indexing::Slice(0, 3)}).squeeze(0);  // (3, H, W)

  if (remove_background) {
    if (channels != 4) {
      throw std::runtime_error(
          "preprocess_image: remove_background=true requires a 4-channel (RGBA) input; "
          "the C++ library does not run BiRefNet. Pre-cutout with scripts/rmbg_preprocess.py.");
    }
    torch::Tensor alpha = chw.index({torch::indexing::Slice(3, 4)});  // (1, 1, H, W)
    // Erode.
    alpha = erode_alpha(alpha, erode_radius);

    // bbox of nonzero alpha (Python: np.nonzero(alpha)).
    auto a_cpu = alpha.squeeze(0).squeeze(0).to(torch::kCPU);  // (H, W)
    auto nonzero = a_cpu.nonzero();  // (K, 2) = (y, x)
    if (nonzero.size(0) == 0) {
      // No foreground — fall back to full frame.
      // Center-pad to square below will handle it.
      torch::Tensor square = rgb.unsqueeze(0);  // (1, 3, H, W)
      int sq = std::max(new_h, new_w);
      torch::Tensor padded = torch::zeros({1, 3, sq, sq}, square.options());
      padded.index_put_({torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         torch::indexing::Slice((sq - new_h) / 2, (sq - new_h) / 2 + new_h),
                         torch::indexing::Slice((sq - new_w) / 2, (sq - new_w) / 2 + new_w)},
                        square);
      return resize_bilinear(padded, canvas_size, canvas_size)
          .to(device, /*non_blocking=*/false);
    }

    auto ys = nonzero.index({torch::indexing::Slice(), 0});
    auto xs = nonzero.index({torch::indexing::Slice(), 1});
    int64_t x_min = xs.min().item<int64_t>();
    int64_t x_max = xs.max().item<int64_t>();
    int64_t y_min = ys.min().item<int64_t>();
    int64_t y_max = ys.max().item<int64_t>();
    double cx = (x_min + x_max) / 2.0;
    double cy = (y_min + y_max) / 2.0;
    double half = std::max(x_max - x_min, y_max - y_min) / 2.0 * 1.2;

    int64_t x0 = static_cast<int64_t>(std::floor(cx - half));
    int64_t y0 = static_cast<int64_t>(std::floor(cy - half));
    int64_t x1 = static_cast<int64_t>(std::ceil(cx + half));
    int64_t y1 = static_cast<int64_t>(std::ceil(cy + half));
    // Clamp to image bounds.
    x0 = std::max<int64_t>(0, x0); y0 = std::max<int64_t>(0, y0);
    x1 = std::min<int64_t>(new_w, x1); y1 = std::min<int64_t>(new_h, y1);
    if (x1 <= x0) x1 = x0 + 1;
    if (y1 <= y0) y1 = y0 + 1;

    // Crop RGB and alpha.
    torch::Tensor rgb_crop = rgb.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(y0, y1),
         torch::indexing::Slice(x0, x1)});  // (3, ch, cw)
    torch::Tensor alpha_crop = alpha.squeeze(0).index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(y0, y1),
         torch::indexing::Slice(x0, x1)});  // (1, ch, cw)

    // Resize to canvas_size x canvas_size.
    torch::Tensor rgb_resized = resize_bilinear(
        rgb_crop.unsqueeze(0), canvas_size, canvas_size).squeeze(0);  // (3, S, S)
    torch::Tensor alpha_resized = resize_bilinear(
        alpha_crop.unsqueeze(0), canvas_size, canvas_size).squeeze(0);  // (1, S, S)

    // Composite foreground on black: out = rgb * alpha.
    torch::Tensor composited = rgb_resized * alpha_resized;  // (3, S, S)
    return composited.unsqueeze(0).to(device, /*non_blocking=*/false);  // (1, 3, S, S)
  } else {
    // Keep background: center-pad to square, then resize.
    int sq = std::max(new_h, new_w);
    torch::Tensor padded = torch::zeros({1, 3, sq, sq}, rgb.options());
    padded.index_put_({torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice((sq - new_h) / 2, (sq - new_h) / 2 + new_h),
                       torch::indexing::Slice((sq - new_w) / 2, (sq - new_w) / 2 + new_w)},
                      rgb.unsqueeze(0));
    return resize_bilinear(padded, canvas_size, canvas_size)
        .to(device, /*non_blocking=*/false);  // (1, 3, S, S)
  }
}

// Helper exposed for the pipeline: load an image file into a (H, W, C) tensor.
torch::Tensor load_image(const std::string& path, int req_channels) {
  return load_image_file(path, req_channels);
}

}  // namespace triposplat::detail
