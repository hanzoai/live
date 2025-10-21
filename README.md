# Hanzo Live

[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/mnfGR4Fjhp)

![longlivedemo1](https://github.com/user-attachments/assets/7fa46227-4405-4ad9-9cd7-a53724f0203d)

Hanzo Live is a tool for running and customizing real-time, interactive generative AI pipelines and models.

🚧 Here be dragons! This project is currently in **alpha**. 🚧

## Features

- Autoregressive video diffusion models
  - [StreamDiffusionV2](./pipelines/streamdiffusionv2/docs/usage.md)
  - [LongLive](./pipelines/longlive/docs/usage.md)
- WebRTC real-time streaming
- Low latency async video processing pipelines
- Interactive UI with text prompting, model parameter controls and video/camera/text input modes

...and more to come!

## System Requirements

Hanzo Live currently supports the following operating systems:

- Linux
- Windows
- macOS (Apple Silicon with MLX support)

### GPU Requirements

**NVIDIA GPUs (Linux/Windows):**
- Requires a Nvidia GPU with >= 24GB VRAM
- We recommend a driver that supports CUDA >= 12.8
- RTX 3090/4090/5090 recommended (newer generations will support higher FPS throughput and lower latency)
- If you do not have access to a GPU with these specs, we recommend installing on [Runpod](#runpod)

**Apple Silicon (macOS):**
- Supported on M1/M2/M3/M4 Macs with unified memory
- Automatically uses MLX (Apple's machine learning framework) with Metal backend
- No special flags needed - Apple Silicon acceleration is auto-detected

## Install

### Manual Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) which is needed to run the server and [Node.js](https://nodejs.org/en/download) which is needed to build the frontend.

#### Clone

```
git clone git@github.com:hanzoai/live.git
cd live
```

#### Build

This will build the frontend files which will be served by the Hanzo Live server.

```
uv run build
```

#### Run

> [!IMPORTANT]
> If you are running the server in a cloud environment, make sure to read the [Firewalls](#firewalls) section.

This will start the server and on the first run will also download required model weights. The default directory where model weights are stored is `~/.hanzo-live/models`.

```bash
uv run hanzo-live
```

The application will automatically detect your hardware:
- **NVIDIA GPU** (Linux/Windows) → Uses CUDA acceleration
- **Apple Silicon** (macOS) → Uses MLX/Metal acceleration
- **CPU fallback** → Use `--cpu` flag for testing without GPU

After the server starts up, the frontend will be available at `http://localhost:8000`.

### Runpod

Use our RunPod template to quickly set up Hanzo Live in the cloud. This is the easiest way to get started if you don't have a compatible local GPU.

> [!IMPORTANT]
> Follow the instructions in [Firewalls](#firewalls) to get a HuggingFace access token.

**Deployment Steps:**

1. **Click the Runpod template link**: [Template](https://console.runpod.io/deploy?template=aca8mw9ivw&ref=5k8hxjq3)

2. **Select your GPU**: Choose a GPU that meets the [system requirements](#system-requirements).

3. **Configure environment variables**:
   - Click "Edit Template"
   - Add an environment variable:
     - Set name to `HF_TOKEN`
     - Set value to your HuggingFace access token
   - Click "Set Overrides"

4. **Deploy**: Click "Deploy On-Demand"

5. **Access the app**: Wait for deployment to complete, then open the app at port 8000

The template will automatically download model weights and configure everything needed.

## Firewalls

If you run Hanzo Live in a cloud environment with restrictive firewall settings (eg. Runpod), Hanzo Live supports using [TURN servers](https://webrtc.org/getting-started/turn-server) to establish a connection between your browser and the streaming server.

The easiest way to enable this feature is to create a HuggingFace account and a `read` [access token](https://huggingface.co/docs/hub/en/security-tokens). You can then set an environment variable before starting Hanzo Live:

```bash
# You should set this to your HuggingFace access token
export HF_TOKEN=your_token_here
```

When you start Hanzo Live, it will automatically use Cloudflare's TURN servers and you'll have 10GB of free streaming per month:

```
uv run hanzo-live
```

## Contributing

Read the [contribution guide](./docs/contributing.md).

## License

The alpha version of this project is licensed under [CC BY-NC-SA 4.0](./LICENSE).

You may use, modify, and share the code for non-commercial purposes only, provided that proper attribution is given.

We will consider re-licensing future versions under a more permissive license if/when non-commercial dependencies are refactored or replaced.

---

Copyright © 2025 Hanzo AI Inc. All rights reserved.
