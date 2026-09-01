// Smoke test for the GaussianIO writers: build a tiny synthetic Gaussian,
// write PLY + SPLAT, then read them back and check the byte layout matches
// the documented format. Does not require TorchScript weights or CUDA.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "triposplat/IO.hpp"
#include "triposplat/Gaussian.hpp"

namespace {

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
}

int run() {
  // 4 synthetic gaussians at known positions.
  int N = 4;
  triposplat::Gaussian g;
  g.means = torch::tensor({{-0.1f, -0.2f, 0.3f},
                           { 0.0f,  0.0f, 0.0f},
                           { 0.5f,  0.5f, 0.5f},
                           { 0.2f, -0.1f, 0.4f}});  // (N,3)
  g.scales = torch::full({N, 3}, 0.01f);
  g.quats = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                           {1.0f, 0.0f, 0.0f, 0.0f},
                           {1.0f, 0.0f, 0.0f, 0.0f},
                           {1.0f, 0.0f, 0.0f, 0.0f}});  // identity raw quats
  g.opacities = torch::full({N, 1}, 0.9f);
  g.sh0 = torch::zeros({N, 1, 3});
  g.sh0.index_put_({1, 0, 0}, 0.5f);  // a nonzero DC for splat RGB
  g.aabb = torch::tensor({{-0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}});

  std::string ply_path = "test_gaussian_io.ply";
  std::string splat_path = "test_gaussian_io.splat";

  if (!triposplat::save_ply(ply_path, g)) { std::printf("FAIL: save_ply\n"); return 1; }
  if (!triposplat::save_splat(splat_path, g)) { std::printf("FAIL: save_splat\n"); return 1; }

  // PLY header check.
  auto ply = read_file(ply_path);
  std::string hdr(ply.begin(), ply.begin() + std::min<size_t>(ply.size(), 600));
  if (hdr.rfind("ply\nformat binary_little_endian", 0) != 0) {
    std::printf("FAIL: PLY header\n"); return 1;
  }
  if (hdr.find("element vertex 4") == std::string::npos) {
    std::printf("FAIL: PLY vertex count\n"); return 1;
  }
  if (hdr.find("end_header\n") == std::string::npos) {
    std::printf("FAIL: PLY end_header\n"); return 1;
  }
  // 17 float properties per vertex * 4 bytes * 4 vertices = 272 bytes of data.
  size_t hdr_end = hdr.find("end_header\n") + std::string("end_header\n").size();
  size_t expected_data = 17 * 4 * N;
  if (ply.size() - hdr_end != expected_data) {
    std::printf("FAIL: PLY data size %zu != %zu\n", ply.size() - hdr_end, expected_data);
    return 1;
  }

  // SPLAT: 32 bytes/splat * N = 128 bytes (no header).
  auto splat = read_file(splat_path);
  if (splat.size() != 32 * N) {
    std::printf("FAIL: SPLAT size %zu != %zu\n", splat.size(), 32 * N);
    return 1;
  }

  std::printf("OK: gaussian_io smoke test passed "
              "(PLY %zu bytes, SPLAT %zu bytes)\n", ply.size(), splat.size());
  std::remove(ply_path.c_str());
  std::remove(splat_path.c_str());
  return 0;
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
}
