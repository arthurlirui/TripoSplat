#include "triposplat/TripoSplat.hpp"

#include "Decoder.hpp"
#include "Encoders.hpp"
#include "FlowSampler.hpp"
#include "Preprocess.hpp"

#include <torch/torch.h>
#include <c10/util/Exception.h>  // c10::Error

#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace triposplat {

namespace {

namespace fs = std::filesystem;

// Validate num_gaussians: must be in [32768, 262144], rounded to a multiple of
// gaussians_per_point. Mirrors TripoSplatPipeline._validate_num_gaussians.
int validate_num_gaussians(int n, int gpp) {
  constexpr int kMin = 32768;
  constexpr int kMax = 262144;
  if (n < kMin || n > kMax) {
    throw TripoSplatError("num_gaussians must be in [" + std::to_string(kMin) +
                          ", " + std::to_string(kMax) + "], got " + std::to_string(n));
  }
  if (n % gpp == 0) return n;
  int rounded = static_cast<int>(std::round(static_cast<double>(n) / gpp)) * gpp;
  return rounded;
}

void ensure_weights_dir(const std::string& dir) {
  if (!fs::is_directory(dir)) {
    throw TripoSplatError("weights_dir does not exist or is not a directory: " + dir);
  }
}

}  // namespace

struct TripoSplatPipeline::Impl {
  TripoSplatConfig cfg;
  std::unique_ptr<detail::Encoders> encoders;
  std::unique_ptr<detail::FlowSampler> flow;
  std::unique_ptr<detail::Decoder> decoder;

  Impl(const TripoSplatConfig& c) : cfg(c) {
    ensure_weights_dir(cfg.weights_dir);
    encoders = std::make_unique<detail::Encoders>(cfg.weights_dir, cfg.device);
    flow = std::make_unique<detail::FlowSampler>(cfg.weights_dir, cfg.device);
    decoder = std::make_unique<detail::Decoder>(cfg.weights_dir, cfg.device);
  }
};

TripoSplatPipeline::TripoSplatPipeline(const TripoSplatConfig& cfg)
    : pimpl_(std::make_unique<Impl>(cfg)) {}

TripoSplatPipeline::~TripoSplatPipeline() = default;
TripoSplatPipeline::TripoSplatPipeline(TripoSplatPipeline&&) noexcept = default;
TripoSplatPipeline& TripoSplatPipeline::operator=(TripoSplatPipeline&&) noexcept = default;

Gaussian TripoSplatPipeline::run(const torch::Tensor& image,
                                 uint64_t seed, int steps,
                                 float guidance_scale, float shift,
                                 int num_gaussians, int erode_radius,
                                 bool remove_background) {
  if (!pimpl_) throw TripoSplatError("pipeline is not initialized");

  const auto& cfg = pimpl_->cfg;
  const auto& device = cfg.device;

  try {
    // Run in no-grad mode (matches Python's @torch.no_grad() on the sampler).
    torch::NoGradGuard no_grad;

    auto* dec = pimpl_->decoder.get();
    int gpp = dec->gaussians_per_point();
    int n = validate_num_gaussians(num_gaussians, gpp);

    // Seed the global CUDA generator once so the VAE's stochastic latent and the
    // flow sampler's initial noise share one advancing stream (matches Python's
    // single torch.Generator seeded with `seed`).
    torch::manual_seed(seed);

    // 1. Preprocess -> (1, 3, canvas, canvas) float32 RGB on device.
    torch::Tensor prepared = detail::preprocess_image(
        image, remove_background, erode_radius, cfg.canvas_size, device);

    // 2. Encode (DinoV3 + Flux2 VAE).
    detail::EncodedImage enc = pimpl_->encoders->encode(prepared, seed);

    // 3. Sample latent (Euler + CFG).
    detail::SampledLatent sampled = pimpl_->flow->sample(
        enc.feature1, enc.feature2, seed, steps, guidance_scale, shift);

    // 4. Decode -> Gaussians.
    return dec->decode(sampled.latent, n);
  } catch (const TripoSplatError&) {
    throw;  // Already the right type; re-raise as-is.
  } catch (const c10::Error& e) {
    throw TripoSplatError(std::string("libtorch error: ") + e.what());
  } catch (const std::runtime_error& e) {
    throw TripoSplatError(std::string("runtime error: ") + e.what());
  }
}

}  // namespace triposplat
