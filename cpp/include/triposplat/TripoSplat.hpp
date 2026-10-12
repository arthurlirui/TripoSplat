#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "triposplat/Export.h"
#include "triposplat/Gaussian.hpp"

namespace triposplat {

// Exception type raised by the pipeline for unrecoverable errors
// (missing weights, bad image, etc.).
class TRIPOSPLAT_API TripoSplatError : public std::runtime_error {
 public:
  explicit TripoSplatError(const std::string& msg) : std::runtime_error(msg) {}
};

// Configuration for TripoSplatPipeline.
struct TripoSplatConfig {
  // Directory containing the TorchScript .pt files produced by
  // scripts/export_torchscript.py:
  //   dinov3.pt, vae_encoder.pt, flow_model.pt, octree.pt, gs_decoder.pt
  std::string weights_dir = "cpp/weights";

  // Compute device. Use torch::kCUDA for GPU inference.
  torch::Device device = torch::kCUDA;

  // Preprocessing canvas size (the pipeline resizes so min(w,h) == canvas_size).
  int canvas_size = 1024;

  // SH constant C0 (matches Python SH_C0 = 0.28209479177387814).
  float sh_c0 = 0.28209479177387814f;
};

// C++ mirror of the Python `TripoSplatPipeline` Stage-A generator.
//
// One image in -> one Gaussian bundle out. The input `image` is expected to
// already have its background removed (RGBA) when `remove_background` is true;
// the C++ library does not run BiRefNet. See scripts/rmbg_preprocess.py.
class TripoSplatPipeline {
 public:
  explicit TripoSplatPipeline(const TripoSplatConfig& cfg = {});
  ~TripoSplatPipeline();

  TripoSplatPipeline(const TripoSplatPipeline&) = delete;
  TripoSplatPipeline& operator=(const TripoSplatPipeline&) = delete;
  TripoSplatPipeline(TripoSplatPipeline&&) noexcept;
  TripoSplatPipeline& operator=(TripoSplatPipeline&&) noexcept;

  // Run the full Stage-A generation.
  //
  //   image: input tensor. Accepted shapes (on any device, will be moved to
  //          cfg.device):
  //            (H, W, 3) uint8  - RGB (requires remove_background=false)
  //            (H, W, 4) uint8  - RGBA (alpha already cut out; use with
  //                               remove_background=true to composite on black)
  //            (1, 3, H, W) float in [0,1] - normalized RGB
  //            (1, 4, H, W) float in [0,1] - normalized RGBA
  //   seed:            RNG seed for VAE latent + flow noise (same seed -> same
  //                    output). Default 42, matching Python.
  //   steps:           flow-matching Euler steps (10-20 recommended).
  //   guidance_scale:  CFG strength (<=1 disables CFG). Default 3.0.
  //   shift:           timestep schedule shift. Default 3.0.
  //   num_gaussians:   target splat count, rounded to a multiple of 32.
  //                    Range [32768, 262144]. Default 262144.
  //   erode_radius:    alpha erosion radius in px (0 disables). Default 1.
  //   remove_background: if true, composite the RGBA foreground on black
  //                    (expects a 4-channel input). If false, center-pad the
  //                    RGB image to square.
  //
  // Returns the assembled Gaussian (see Gaussian.hpp). Throws TripoSplatError
  // on failure.
  Gaussian run(const torch::Tensor& image,
               uint64_t seed = 42,
               int steps = 20,
               float guidance_scale = 3.0f,
               float shift = 3.0f,
               int num_gaussians = 262144,
               int erode_radius = 1,
               bool remove_background = true);

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace triposplat
