---
license: mit
---

# TripoSplat: Generative 3D Gaussians with Learned Density Control

TripoSplat converts a single 2D image into high-quality and variable number of 3D Gaussians developed by [TripoAI](https://www.tripo3d.ai/). It can serve as a powerful pipeline tool for asset creation, AR/VR, game development, simulation environments, and beyond.

[Read Paper](https://arxiv.org/abs/2605.16355) | [Technical Blog](https://www.tripo3d.ai/research/triposplat)

# Usage
To use with the [official inference code](https://github.com/VAST-AI-Research/TripoSplat), place the files in the following structure:
```
📂 TripoSplat/
├── 📂 ckpts/
│   ├── 📂 background_removal/
│   │   └── birefnet.safetensors
│   ├── 📂 diffusion_models/
│   │   └── triposplat_fp16.safetensors
│   ├── 📂 clip_vision/
│   │   └── dino_v3_vit_h.safetensors
│   ├── 📂 vae/
│   │   ├── flux2-vae.safetensors
│   │   └── triposplat_vae_decoder_fp16.safetensors
```

To use with [ComfyUI](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/3d_triposplat_image_to_gaussian_splat.json), place the files in the following structure:
```
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 background_removal/
│   │   └── birefnet.safetensors
│   ├── 📂 diffusion_models/
│   │   └── triposplat_fp16.safetensors
│   ├── 📂 clip_vision/
│   │   └── dino_v3_vit_h.safetensors
│   ├── 📂 vae/
│   │   ├── flux2-vae.safetensors
│   │   └── triposplat_vae_decoder_fp16.safetensors
```
