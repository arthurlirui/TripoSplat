// End-to-end Stage-A demo: load an RGBA cutout image -> run the C++ TripoSplat
// pipeline -> write gaussians.ply and gaussians.splat.
//
// Prerequisites:
//   1. python scripts/export_torchscript.py --out cpp/weights
//   2. python scripts/rmbg_preprocess.py <input.jpg> <input_rgba.png>
//      (the C++ library does not run BiRefNet; feed it an RGBA cutout)
//
// Usage:
//   run_example <rgba_image.png> [output_dir] [--weights cpp/weights]
//                              [--seed 42] [--steps 20] [--num-gaussians 262144]
//                              [--device cuda]
//
// If the input is a 3-channel RGB image, pass --no-remove-bg to skip the
// alpha-composite path (the image will be center-padded to square instead).

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#include <torch/torch.h>
#include <stb_image.h>

#include "triposplat/IO.hpp"
#include "triposplat/TripoSplat.hpp"

namespace fs = std::filesystem;

struct Args {
  std::string image_path;
  std::string output_dir = ".";
  std::string weights_dir = "cpp/weights";
  std::string device = "cuda";
  uint64_t seed = 42;
  int steps = 20;
  int num_gaussians = 262144;
  bool remove_bg = true;
};

void usage() {
  std::cerr << "Usage: run_example <rgba_image.png> [output_dir] [--weights cpp/weights] "
               "[--device cuda] [--seed 42] [--steps 20] [--num-gaussians 262144] "
               "[--no-remove-bg]\n";
}

Args parse_args(int argc, char** argv) {
  Args a;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      return argv[++i];
    };
    if (s == "--weights") a.weights_dir = next();
    else if (s == "--device") a.device = next();
    else if (s == "--seed") a.seed = std::stoull(next());
    else if (s == "--steps") a.steps = std::stoi(next());
    else if (s == "--num-gaussians") a.num_gaussians = std::stoi(next());
    else if (s == "--no-remove-bg") a.remove_bg = false;
    else if (s == "--remove-bg") a.remove_bg = true;
    else if (s == "-h" || s == "--help") { usage(); std::exit(0); }
    else if (s.substr(0, 2) == "--") { usage(); std::exit(2); }
    else positional.push_back(s);
  }
  if (positional.empty()) { usage(); std::exit(2); }
  a.image_path = positional[0];
  if (positional.size() > 1) a.output_dir = positional[1];
  return a;
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  // Load the image via stb_image through the pipeline's helper.
  // We request 4 channels when removing the background (RGBA cutout), else 3.
  // Build the (H, W, C) uint8 tensor the pipeline accepts.
  auto load_to_tensor = [](const std::string& path, int req_c) -> torch::Tensor {
    // Use stb_image directly here to avoid leaking the detail:: symbol.
    int w = 0, h = 0, c = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, req_c);
    if (!data) {
      std::cerr << "Failed to load image '" << path << "': " << stbi_failure_reason() << "\n";
      std::exit(1);
    }
    int channels = req_c ? req_c : c;
    auto opts = torch::TensorOptions().dtype(torch::kUInt8);
    torch::Tensor img = torch::from_blob(data, {h, w, channels}, opts).clone();
    stbi_image_free(data);
    return img;
  };

  torch::Tensor image = load_to_tensor(
      args.image_path, args.remove_bg ? 4 : 3);
  std::cout << "[run_example] loaded " << args.image_path
            << " (" << image.size(0) << "x" << image.size(1)
            << "x" << image.size(2) << ")\n";

  triposplat::TripoSplatConfig cfg;
  cfg.weights_dir = args.weights_dir;
  cfg.device = torch::Device(args.device);

  std::cout << "[run_example] initializing pipeline (weights=" << cfg.weights_dir
            << ", device=" << args.device << ") ...\n";
  triposplat::TripoSplatPipeline pipeline(cfg);

  std::cout << "[run_example] running (seed=" << args.seed
            << ", steps=" << args.steps
            << ", num_gaussians=" << args.num_gaussians
            << ", remove_bg=" << (args.remove_bg ? "true" : "false") << ") ...\n";
  triposplat::Gaussian g = pipeline.run(
      image, args.seed, args.steps, /*guidance_scale=*/3.0f, /*shift=*/3.0f,
      args.num_gaussians, /*erode_radius=*/1, args.remove_bg);

  std::cout << "[run_example] generated " << g.size() << " gaussians\n";

  fs::create_directories(args.output_dir);
  std::string ply_path = (fs::path(args.output_dir) / "gaussians.ply").string();
  std::string splat_path = (fs::path(args.output_dir) / "gaussians.splat").string();

  if (triposplat::save_ply(ply_path, g)) {
    std::cout << "[run_example] wrote " << ply_path << "\n";
  } else {
    std::cerr << "[run_example] FAILED to write " << ply_path << "\n";
  }
  if (triposplat::save_splat(splat_path, g)) {
    std::cout << "[run_example] wrote " << splat_path << "\n";
  } else {
    std::cerr << "[run_example] FAILED to write " << splat_path << "\n";
  }

  return 0;
}
