<div align="center">
<img src="https://github.com/vladmandic/sdnext/raw/master/html/logo-transparent.png" width=200 alt="SD.Next: AI art generator logo">

# SD.Next: All-in-one WebUI

SD.Next is a powerful, open-source WebUI app for AI image and video generation, built on Stable Diffusion and supporting dozens of advanced models. Create, caption, and process images and videos with a modern, cross-platform interface—perfect for artists, researchers, and AI enthusiasts.


![Stars](https://img.shields.io/github/stars/vladmandic/sdnext?style=social)
![Forks](https://img.shields.io/github/forks/vladmandic/sdnext?style=social)
![Contributors](https://img.shields.io/github/contributors/vladmandic/sdnext)
![Last update](https://img.shields.io/github/last-commit/vladmandic/sdnext?svg=true)
![License](https://img.shields.io/github/license/vladmandic/sdnext?svg=true)
[![Discord](https://img.shields.io/discord/1101998836328697867?logo=Discord&svg=true)](https://discord.gg/VjvR2tabEX)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/vladmandic/sdnext)
[![Sponsors](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/vladmandic)

[Docs](https://vladmandic.github.io/sdnext-docs/) | [Wiki](https://github.com/vladmandic/sdnext/wiki) | [Discord](https://discord.gg/VjvR2tabEX) | [Changelog](CHANGELOG.md)

</div>
</br>

## Table of contents

- [Documentation](https://vladmandic.github.io/sdnext-docs/)
- [SD.Features](#features--capabilities)
- [Supported AI Models](#supported-ai-models)
- [Supported Platforms & Hardware](#supported-platforms--hardware)
- [Getting started](#getting-started)

### Screenshot: Desktop interface

<div align="center">
<img src="https://github.com/vladmandic/sdnext/raw/dev/html/screenshot-robot.jpg" alt="SD.Next: AI art generator desktop interface screenshot" width="90%">
</div>

### Screenshot: Mobile interface

<div align="center">
<img src="https://github.com/user-attachments/assets/ced9fe0c-d2c2-46d1-94a7-8f9f2307ce38" alt="SD.Next: AI art generator mobile interface screenshot" width="35%">
</div>
</div>

<br>

## Features & Capabilities

SD.Next is feature-rich with a focus on performance, flexibility, and user experience. Key features include:
- [Multi-platform](#platform-support!  
- Many [diffusion models](https://vladmandic.github.io/sdnext-docs/Model-Support/)!  
- Fully localized to ~15 languages and with support for many [UI themes](https://vladmandic.github.io/sdnext-docs/Themes/)!
- [Desktop](#screenshot-desktop-interface) and [Mobile](#screenshot-mobile-interface) support!  
- Platform specific auto-detection and tuning performed on install  
- Built in installer with automatic updates and dependency management  

### Unique features

SD.Next includes many features not found in other WebUIs, such as:
- **SDNQ**: State-of-the-Art quantization engine
  Use pre-quantized or run with quantizaion on-the-fly for up to 4x VRAM reduction with no or minimal quality and performance impact  
- **Balanced Offload**: Dynamically balance CPU and GPU memory to run larger models on limited hardware
- **Captioning** with 150+ **OpenCLiP** models, **Tagger** with **WaifuDiffusion** and **DeepDanbooru** models, and 25+ built-in **VLMs**  
- **Image Processing** with full image correction color-grading suite of tools  

<br>

## Supported AI Models

SD.Next supports broad range of models: [supported models](https://vladmandic.github.io/sdnext-docs/Model-Support/) and [model specs](https://vladmandic.github.io/sdnext-docs/Models/)  

## Supported Platforms & Hardware

- *nVidia* GPUs using **CUDA** libraries on both *Windows and Linux*  
- *AMD* GPUs using **ROCm** libraries on both *Linux and Windows*
- *AMD* GPUs on Windows using **ZLUDA** libraries  
- *Intel Arc* GPUs using **OneAPI** with *IPEX XPU* libraries on both *Windows and Linux*  
- Any *CPU/GPU* or device compatible with **OpenVINO** libraries on both *Windows and Linux*  
- Any GPU compatible with *DirectX* on *Windows* using **DirectML** libraries  
- *Apple M1/M2* on *OSX* using built-in support in Torch with **MPS** optimizations  
- *ONNX/Olive*  

Plus **Docker** container recipes for: [CUDA, ROCm, Intel IPEX and OpenVINO](https://vladmandic.github.io/sdnext-docs/Docker/)

## Getting started

- Get started with **SD.Next** by following the [installation instructions](https://vladmandic.github.io/sdnext-docs/Installation/)  
- For more details, check out [advanced installation](https://vladmandic.github.io/sdnext-docs/Advanced-Install/) guide  
- List and explanation of [command line arguments](https://vladmandic.github.io/sdnext-docs/CLI-Arguments/)  
- Install walkthrough [video](https://www.youtube.com/watch?v=nWTnTyFTuAs)  

> [!TIP]
> And for platform specific information, check out  
> [WSL](https://vladmandic.github.io/sdnext-docs/WSL/) | [Intel Arc](https://vladmandic.github.io/sdnext-docs/Intel-ARC/) | [DirectML](https://vladmandic.github.io/sdnext-docs/DirectML/) | [OpenVINO](https://vladmandic.github.io/sdnext-docs/OpenVINO/) | [ONNX & Olive](https://vladmandic.github.io/sdnext-docs/ONNX-Runtime/) | [ZLUDA](https://vladmandic.github.io/sdnext-docs/ZLUDA/) | [AMD ROCm](https://vladmandic.github.io/sdnext-docs/AMD-ROCm/) | [MacOS](https://vladmandic.github.io/sdnext-docs/MacOS-Python/) | [nVidia](https://vladmandic.github.io/sdnext-docs/nVidia/) | [Docker](https://vladmandic.github.io/sdnext-docs/Docker/)

### Quick Start

```shell
git clone https://github.com/vladmandic/sdnext
cd sdnext
./webui.sh # Linux/Mac
webui.bat  # Windows
webui.ps1  # PowerShell
```

> [!WARNING]
> If you run into issues, check out [troubleshooting](https://vladmandic.github.io/sdnext-docs/Troubleshooting/) and [debugging](https://vladmandic.github.io/sdnext-docs/Debug/) guides  


## Community & Support

If you're unsure how to use a feature, best place to start is [Docs](https://vladmandic.github.io/sdnext-docs/) and if its not there,  
check [ChangeLog](https://vladmandic.github.io/sdnext-docs/CHANGELOG/) for when feature was first introduced as it will always have a short note on how to use it  

And for any question, reach out on [Discord](https://discord.gg/VjvR2tabEX) or open an [issue](https://github.com/vladmandic/sdnext/issues) or [discussion](https://github.com/vladmandic/sdnext/discussions)  

### Contributing

Please see [Contributing](CONTRIBUTING) for details on how to contribute to this project  

## License & Credits

- SD.Next is licensed under the [Apache License 2.0](LICENSE.txt)
- Main credit goes to [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for the original codebase  

## Evolution

<a href="https://star-history.com/#vladmandic/sdnext&Date">
  <picture width=640>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=vladmandic/sdnext&type=Date&theme=dark" />
    <img src="https://api.star-history.com/svg?repos=vladmandic/sdnext&type=Date" alt="starts" width="320">
  </picture>
</a>

- [OSS Stats](https://ossinsight.io/analyze/vladmandic/sdnext#overview)

<br>
