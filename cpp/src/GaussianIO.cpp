#include "triposplat/Gaussian.hpp"
#include "triposplat/IO.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace triposplat {

namespace {

// Default TripoSplat -> Y-up world transform applied to means and quaternions
// (matches Gaussian._DEFAULT_TRANSFORM in triposplat.py).
//   [[1, 0, 0],
//    [0, 0,-1],
//    [0, 1, 0]]
constexpr float kDefaultTransform[3][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, -1.0f},
    {0.0f, 1.0f, 0.0f},
};

// Inverse-opacity activation: logit(x) = log(x / (1 - x)).
// Mirrors Gaussian._inverse_opacity_activation.
inline float inverse_opacity(float x) {
  x = std::clamp(x, 1e-6f, 1.0f - 1e-6f);
  return std::log(x / (1.0f - x));
}

// Convert a raw (w, x, y, z) quaternion to a 3x3 rotation matrix.
inline void quat_to_matrix(const float q[4], float R[9]) {
  float w = q[0], x = q[1], y = q[2], z = q[3];
  // Normalize to unit length (Python normalizes before matrix_to_quat calls).
  float n = std::sqrt(w * w + x * x + y * y + z * z);
  if (n < 1e-12f) {
    R[0] = 1; R[1] = 0; R[2] = 0;
    R[3] = 0; R[4] = 1; R[5] = 0;
    R[6] = 0; R[7] = 0; R[8] = 1;
    return;
  }
  w /= n; x /= n; y /= n; z /= n;
  R[0] = 1 - 2 * (y * y + z * z);
  R[1] = 2 * (x * y - z * w);
  R[2] = 2 * (x * z + y * w);
  R[3] = 2 * (x * y + z * w);
  R[4] = 1 - 2 * (x * x + z * z);
  R[5] = 2 * (y * z - x * w);
  R[6] = 2 * (x * z - y * w);
  R[7] = 2 * (y * z + x * w);
  R[8] = 1 - 2 * (x * x + y * y);
}

// Convert a 3x3 rotation matrix back to a (w, x, y, z) quaternion using the
// Shepperd 4-branch method (matches multiview_refine/quaternion.py matrix_to_quat).
inline void matrix_to_quat(const float R[9], float q[4]) {
  float tr = R[0] + R[4] + R[8];
  float qw, qx, qy, qz;
  if (tr > 0.0f) {
    float s = std::sqrt(tr + 1.0f) * 2.0f;  // s = 4 * qw
    qw = 0.25f * s;
    qx = (R[7] - R[5]) / s;
    qy = (R[2] - R[6]) / s;
    qz = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    float s = std::sqrt(1.0f + R[0] - R[4] - R[8]) * 2.0f;  // s = 4 * qx
    qw = (R[7] - R[5]) / s;
    qx = 0.25f * s;
    qy = (R[1] + R[3]) / s;
    qz = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    float s = std::sqrt(1.0f + R[4] - R[0] - R[8]) * 2.0f;  // s = 4 * qy
    qw = (R[2] - R[6]) / s;
    qx = (R[1] + R[3]) / s;
    qy = 0.25f * s;
    qz = (R[5] + R[7]) / s;
  } else {
    float s = std::sqrt(1.0f + R[8] - R[0] - R[4]) * 2.0f;  // s = 4 * qz
    qw = (R[3] - R[1]) / s;
    qx = (R[2] + R[6]) / s;
    qy = (R[5] + R[7]) / s;
    qz = 0.25f * s;
  }
  q[0] = qw; q[1] = qx; q[2] = qy; q[3] = qz;
}

// Apply the default transform to a point: xyz_world = xyz @ T.T.
inline void transform_point(const float xyz[3], float out[3]) {
  for (int i = 0; i < 3; ++i) {
    out[i] = kDefaultTransform[i][0] * xyz[0]
           + kDefaultTransform[i][1] * xyz[1]
           + kDefaultTransform[i][2] * xyz[2];
  }
}

// Apply the default transform to a quaternion: R_world = T @ R_quat, then
// convert back. Mirrors `_transformed_xyz_rot` / `_get_ply_data`.
// `q_in` is the raw predicted quaternion; Python adds the fixed `rots_bias`
// = [1, 0, 0, 0] (identity) before converting to a matrix, so we do the same
// here to match the stored PLY/SPLAT rotation byte-for-byte.
inline void transform_quat(const float q_in[4], float q_out[4]) {
  float qb[4] = {q_in[0] + 1.0f, q_in[1], q_in[2], q_in[3]};  // + rots_bias [1,0,0,0]
  float R[9];
  quat_to_matrix(qb, R);
  // R_world = T @ R
  float Rt[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Rt[i * 3 + j] = kDefaultTransform[i][0] * R[j]
                    + kDefaultTransform[i][1] * R[3 + j]
                    + kDefaultTransform[i][2] * R[6 + j];
    }
  }
  matrix_to_quat(Rt, q_out);
}

// Write a little-endian float32 to the stream (host is x86 little-endian on
// Windows, but this keeps the code portable).
inline void write_f32(std::ostream& os, float v) {
  std::int32_t bits;
  std::memcpy(&bits, &v, 4);
  char buf[4] = {
      static_cast<char>(bits & 0xFF),
      static_cast<char>((bits >> 8) & 0xFF),
      static_cast<char>((bits >> 16) & 0xFF),
      static_cast<char>((bits >> 24) & 0xFF)};
  os.write(buf, 4);
}

}  // namespace

bool save_ply(const std::string& path, const Gaussian& g) {
  int64_t n = g.size();
  if (n <= 0) return false;

  // Move everything to CPU float32 contiguous.
  auto means = g.means.to(torch::kCPU, torch::kFloat, /*non_blocking=*/false).contiguous();
  auto scales_act = g.scales.to(torch::kCPU, torch::kFloat).contiguous();
  auto quats_raw = g.quats.to(torch::kCPU, torch::kFloat).contiguous();
  auto opac_act = g.opacities.to(torch::kCPU, torch::kFloat).contiguous();
  auto sh0 = g.sh0.to(torch::kCPU, torch::kFloat).contiguous();

  // sh0 shape is (N, 1, 3). Flatten to (N, 3) as f_dc_0..2 (Python transposes
  // (1,2) then flattens from dim 1, which for (N,1,3) yields the same order).
  if (sh0.dim() == 3 && sh0.size(1) == 1 && sh0.size(2) == 3) {
    sh0 = sh0.reshape({n, 3});
  } else if (sh0.dim() == 2 && sh0.size(1) == 3) {
    // already (N, 3)
  } else {
    return false;
  }

  auto m_acc = means.accessor<float, 2>();
  auto s_acc = scales_act.accessor<float, 2>();
  auto q_acc = quats_raw.accessor<float, 2>();
  auto o_acc = opac_act.accessor<float, 2>();  // (N, 1)
  auto sh_acc = sh0.accessor<float, 2>();

  std::ofstream f(path, std::ios::binary);
  if (!f) return false;

  // Header (matches _binary_ply_bytes + construct_list_of_attributes).
  std::ostringstream hdr;
  hdr << "ply\nformat binary_little_endian 1.0\n";
  hdr << "element vertex " << n << "\n";
  // x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0..3
  hdr << "property float x\n";
  hdr << "property float y\n";
  hdr << "property float z\n";
  hdr << "property float nx\n";
  hdr << "property float ny\n";
  hdr << "property float nz\n";
  hdr << "property float f_dc_0\n";
  hdr << "property float f_dc_1\n";
  hdr << "property float f_dc_2\n";
  hdr << "property float opacity\n";
  hdr << "property float scale_0\n";
  hdr << "property float scale_1\n";
  hdr << "property float scale_2\n";
  hdr << "property float rot_0\n";
  hdr << "property float rot_1\n";
  hdr << "property float rot_2\n";
  hdr << "property float rot_3\n";
  hdr << "end_header\n";
  f << hdr.str();

  for (int64_t i = 0; i < n; ++i) {
    float xyz[3] = {m_acc[i][0], m_acc[i][1], m_acc[i][2]};
    float xyz_w[3];
    transform_point(xyz, xyz_w);

    // normals = 0
    write_f32(f, xyz_w[0]); write_f32(f, xyz_w[1]); write_f32(f, xyz_w[2]);
    write_f32(f, 0.0f); write_f32(f, 0.0f); write_f32(f, 0.0f);

    // f_dc (no transform)
    write_f32(f, sh_acc[i][0]); write_f32(f, sh_acc[i][1]); write_f32(f, sh_acc[i][2]);

    // opacity raw = logit(activated)
    write_f32(f, inverse_opacity(o_acc[i][0]));

    // scale raw = log(activated)  (Python writes torch.log(get_scaling))
    write_f32(f, std::log(std::max(s_acc[i][0], 1e-12f)));
    write_f32(f, std::log(std::max(s_acc[i][1], 1e-12f)));
    write_f32(f, std::log(std::max(s_acc[i][2], 1e-12f)));

    // rotation: transform the quaternion through the default transform,
    // then write (w, x, y, z). The rots_bias [1,0,0,0] is added inside
    // transform_quat (q_in[0] + 1.0f) before the matrix conversion, matching
    // the Python pipeline's `_get_ply_data`.
    float q_in[4] = {q_acc[i][0], q_acc[i][1], q_acc[i][2], q_acc[i][3]};
    float q_out[4];
    transform_quat(q_in, q_out);
    write_f32(f, q_out[0]); write_f32(f, q_out[1]);
    write_f32(f, q_out[2]); write_f32(f, q_out[3]);
  }

  return static_cast<bool>(f);
}

bool save_splat(const std::string& path, const Gaussian& g) {
  int64_t n = g.size();
  if (n <= 0) return false;

  auto means = g.means.to(torch::kCPU, torch::kFloat).contiguous();
  auto scales_act = g.scales.to(torch::kCPU, torch::kFloat).contiguous();
  auto quats_raw = g.quats.to(torch::kCPU, torch::kFloat).contiguous();
  auto opac_act = g.opacities.to(torch::kCPU, torch::kFloat).contiguous();
  auto sh0 = g.sh0.to(torch::kCPU, torch::kFloat).contiguous();
  if (sh0.dim() == 3) sh0 = sh0.reshape({n, 3});

  // Build the per-splat sort key: opacity[:, 0] * prod(scale).
  auto m_acc = means.accessor<float, 2>();
  auto s_acc = scales_act.accessor<float, 2>();
  auto q_acc = quats_raw.accessor<float, 2>();
  auto o_acc = opac_act.accessor<float, 2>();
  auto sh_acc = sh0.accessor<float, 2>();

  std::vector<float> keys(n);
  std::vector<int64_t> order(n);
  for (int64_t i = 0; i < n; ++i) {
    keys[i] = o_acc[i][0] * (s_acc[i][0] * s_acc[i][1] * s_acc[i][2]);
    order[i] = i;
  }
  // Descending sort (Python: np.argsort(-...)).
  std::sort(order.begin(), order.end(),
            [&](int64_t a, int64_t b) { return keys[a] > keys[b]; });

  std::ofstream f(path, std::ios::binary);
  if (!f) return false;

  constexpr float kC0 = 0.28209479177387814f;
  std::vector<char> buf;
  buf.reserve(static_cast<size_t>(n) * 32);

  auto push_f32 = [&](float v) {
    std::int32_t bits;
    std::memcpy(&bits, &v, 4);
    buf.push_back(static_cast<char>(bits & 0xFF));
    buf.push_back(static_cast<char>((bits >> 8) & 0xFF));
    buf.push_back(static_cast<char>((bits >> 16) & 0xFF));
    buf.push_back(static_cast<char>((bits >> 24) & 0xFF));
  };
  auto push_u8 = [&](unsigned char v) { buf.push_back(static_cast<char>(v)); };

  for (int64_t idx : order) {
    // xyz (transformed)
    float xyz[3] = {m_acc[idx][0], m_acc[idx][1], m_acc[idx][2]};
    float xyz_w[3];
    transform_point(xyz, xyz_w);
    push_f32(xyz_w[0]); push_f32(xyz_w[1]); push_f32(xyz_w[2]);
    // scale (activated, no transform)
    push_f32(s_acc[idx][0]); push_f32(s_acc[idx][1]); push_f32(s_acc[idx][2]);
    // rgba: rgb from SH DC, a from opacity
    float r = std::clamp((sh_acc[idx][0] * kC0 + 0.5f) * 255.0f, 0.0f, 255.0f);
    float gg = std::clamp((sh_acc[idx][1] * kC0 + 0.5f) * 255.0f, 0.0f, 255.0f);
    float b = std::clamp((sh_acc[idx][2] * kC0 + 0.5f) * 255.0f, 0.0f, 255.0f);
    float a = std::clamp(o_acc[idx][0] * 255.0f, 0.0f, 255.0f);
    push_u8(static_cast<unsigned char>(r));
    push_u8(static_cast<unsigned char>(gg));
    push_u8(static_cast<unsigned char>(b));
    push_u8(static_cast<unsigned char>(a));
    // rot: normalize quat, transform through default, then uint8(q*128+128).
    float q_in[4] = {q_acc[idx][0], q_acc[idx][1], q_acc[idx][2], q_acc[idx][3]};
    float q_out[4];
    transform_quat(q_in, q_out);
    for (int k = 0; k < 4; ++k) {
      float v = std::clamp(q_out[k] * 128.0f + 128.0f, 0.0f, 255.0f);
      push_u8(static_cast<unsigned char>(v));
    }
  }

  f.write(buf.data(), static_cast<std::streamsize>(buf.size()));
  return static_cast<bool>(f);
}

}  // namespace triposplat
