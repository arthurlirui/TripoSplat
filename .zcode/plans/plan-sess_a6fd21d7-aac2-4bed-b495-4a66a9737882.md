
# "GS Capture" — iOS 本地3D高斯泼溅重建 App 设计方案

## 背景分析

当前项目的 TripoSplat 生成式流水线模型总量约 **15亿+参数** (DinoV3ViT ~632M + Flux2VAE ~100M + BiRefNet ~200M + FlowModel ~300M + Decoder ~200M)，加权文件共 **3.6GB**，**完全无法在手机上运行**。

但项目中 `multiview_refine/` 目录的多视角优化路径（gsplat 微分光栅化 + 自适应密度控制）提供了正确方向。结合 iOS 独有的 ARKit（实时6-DOF相机位姿追踪），我们可以在本地实现一个完整的 Gaussian Splatting 训练流水线。

**核心思路**：ARKit 替代 COLMAP → 手机围绕物体拍摄 → 在设备上用 Metal 进行 3DGS 可微分训练 → 实时渲染展示。

---

## 实现计划

### 步骤 1：创建项目基础结构

创建 `GSCapture/` 目录下的所有 Swift 项目骨架文件：
- `GSCapture.xcodeproj` 项目配置
- `Sources/App/GSCaptureApp.swift` — SwiftUI App 入口 + 路由（CaptureView / TrainingView / ViewerScreen 三个 Tab）
- `Sources/App/AppState.swift` — ObservableObject 全局状态管理（录制中/训练中/训练进度/GaussianModel 引用）

### 步骤 2：实现数据采集模块（CaptureEngine）

**文件**: `Sources/Capture/CaptureEngine.swift`, `CaptureSessionManager.swift`, `FrameData.swift`

- 使用 `AVCaptureSession` 获取 RGB 视频流（1080p, 30fps）
- 同时运行 `ARSession`（`.worldTracking` 配置）获取 6-DOF 相机位姿
- `FrameData` 结构体：存储 `CVPixelBuffer` + `simd_float4x4` 位姿 + 可选 `CVPixelBuffer` LiDAR 深度图 + 时间戳
- 用户围绕物体走动拍摄，App 引导"请围绕物体缓慢移动"
- 采集完成后基于位姿覆盖度启发式选择 ~50-200 个关键帧（确保视角均匀分布）

### 步骤 3：实现分割模块（SegmentationEngine）

**文件**: `Sources/Segmentation/SegmentationEngine.swift`, `SegmentationResult.swift`

- 封装 Apple Vision Framework：
  - iOS 17+: `VNGenerateForegroundInstanceMaskRequest` → 输出实例遮罩 + Alpha Matte
  - 备选：`VNGeneratePersonSegmentationRequest`（人物主体）
  - 通用回退：`VNGenerateSaliencyImageRequest`
- 对每个关键帧生成 Alpha Matte → 训练时用于背景剔除

### 步骤 4：实现场景初始化模块（SceneInitializer）

**文件**: `Sources/Reconstruction/SceneInitializer.swift`

- 从 ARKit 采集数据初始化稀疏点云：
  - LiDAR 设备（iPhone Pro）：深度图反投影 + 位姿融合 → 密集彩色点云
  - 非 LiDAR 设备：使用 ARKit 特征点 + 多视角三角测量
- 点云下采样到 ~50k-200k 点（作为 3DGS 初始位置）
- 坐标系对齐 + 尺度归一化到 `[-0.5, 0.5]` AABB

### 步骤 5：实现高斯模型存储（GaussianModel）

**文件**: `Sources/Reconstruction/GaussianModel.swift`

- Swift 数据结构管理高斯参数：positions, scales, quaternions, opacities, colors
- 辅助方法：addGaussian(), removeGaussian(), cloneGaussian(), splitGaussian()
- `toMTLBuffer()` — 将数据复制到 Metal Buffer 供 GPU 使用（fp16 格式节省内存）

### 步骤 6：实现 Metal 渲染器（前向 + 反向）

**文件**: `Sources/Metal/Shaders.metal`, `Rasterizer.metal`, `SortAndTile.metal`

最核心部分 — 实现类 gsplat 的 Tile-based 高斯光栅化器：

**前向传播**: 投影高斯到屏幕 → GPU Radix Sort 按深度排序 → Tile-based Alpha Blending → 渲染图像

**反向传播**: 从 loss 梯度出发 → 沿混合路径逐 tile 反传 → d_positions, d_scales, d_quaternions, d_opacities, d_colors

**简化策略**: fp16 计算、仅 DC 颜色（不做完整 SH）、先实现前向再逐步加入反向

### 步骤 7：实现 SSIM Loss（Metal Kernel）

**文件**: `Sources/Metal/SSIM.metal`

- 在 Metal 中实现 SSIM：11x11 高斯模糊（可分离卷积优化）→ 均值/方差/协方差 → SSIM per-pixel
- 正向：计算 SSIM 值 / 反向：SSIM 对每个像素的梯度

### 步骤 8：实现训练循环（GSTrainer + AdaptiveDensity + Optimizer）

**文件**: `Sources/Reconstruction/GSTrainer.swift`, `AdaptiveDensity.swift`, `Optimizer.swift`

训练循环 ~7k-30k iterations：
- 随机选择训练视角 → 前向渲染 → loss = 0.8*L1 + 0.2*(1-SSIM)（仅前景区域）
- 反向传播 → Adam 优化器更新
- 每 100 步: 自适应密度控制 (clone/split/prune)
- 每 3000 步: 不透明度重置

### 步骤 9：实现训练 UI 界面（TrainingView）

**文件**: `Sources/UI/TrainingView.swift`

- SwiftUI 界面：实时预览渲染 vs GT 对比、训练进度条、PSNR 曲线（Swift Charts）
- 训练在后台 DispatchQueue 运行，UI 不阻塞

### 步骤 10：实现 3D 查看器（GSViewer3D + ARGSViewer）

**文件**: `Sources/Renderer/GSRenderer.swift`, `Sources/UI/ViewerScreen.swift`

- 使用 Metal 渲染高斯模型到 `MTKView`
- 手势：单指旋转、双指缩放、双指平移
- AR 模式：高斯模型叠加到 RealityKit ARView

### 步骤 11：实现模型导出（PLY + SPLAT）

**文件**: `Sources/Export/PLYExporter.swift`, `SPLATExporter.swift`

- PLY：标准 binary little-endian 格式（兼容 SuperSplat/SparkJS）
- SPLAT：32-byte-per-splat 格式

### 步骤 12：实现采集 UI（CaptureView）

**文件**: `Sources/UI/CaptureView.swift`

- 实时摄像头预览 + ARKit 追踪状态
- 录制按钮 + 关键帧计数 + 视觉引导提示

### 步骤 13：编译验证与测试

- 单元测试：投影矩阵正确性、Alpha blending 正确性、SSIM 与 PyTorch 对比
- Xcode 中验证项目可编译

---

## 技术栈

| 层级 | 技术 |
|---|---|
| UI | SwiftUI + UIKit |
| 相机 | AVFoundation |
| 位姿追踪 | ARKit worldTracking |
| 分割 | Vision Framework (iOS 17+) |
| GPU 计算 | Metal 3 (compute + render pipeline) |
| 数学库 | Accelerate / simd |
| 最低系统 | iOS 17.0+, A15 Bionic+, 6GB+ RAM |
| 推荐设备 | iPhone 15 Pro / 16 Pro |
