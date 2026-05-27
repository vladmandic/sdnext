<div align="center">
<img src="https://github.com/vladmandic/sdnext/raw/dev/ui/assets/logo-transparent.png" width=200 alt="SD.Next: AI art generator logo">

# SD.Next: All-in-one WebUI

SD.Next is a state-of-the-art, open-source server application and web interface (WebUI) for AI image and video generation, built on Stable Diffusion and supporting dozens of advanced models. Create, caption, and process images and videos with a modern, cross-platform interface — perfect for artists, researchers, and AI enthusiasts.

[![Stars](https://img.shields.io/github/stars/vladmandic/sdnext?style=for-the-badge&color=%237DD3FC)](https://ossinsight.io/analyze/vladmandic/sdnext#overview)
[![Forks](https://img.shields.io/github/forks/vladmandic/sdnext?style=for-the-badge&color=%2360A5FA)](https://github.com/vladmandic/sdnext/forks)
[![Contributors](https://img.shields.io/github/contributors/vladmandic/sdnext?style=for-the-badge&color=%233B82F6)](https://github.com/vladmandic/sdnext/graphs/contributors)
[![License](https://img.shields.io/github/license/vladmandic/sdnext?style=for-the-badge&color=%232563EB)](LICENSE.txt)

![Last release](https://img.shields.io/github/v/tag/vladmandic/sdnext?style=for-the-badge&color=%231E40AF)
![Last commit](https://img.shields.io/github/last-commit/vladmandic/sdnext?style=for-the-badge&color=%231D4ED8)
[![Dev](https://img.shields.io/github/commits-difference/vladmandic/sdnext?base=master&head=dev&style=for-the-badge&color=%231E3A8A)](https://github.com/vladmandic/sdnext/compare/master...dev)

[![Home](https://img.shields.io/badge/Home-teal?style=for-the-badge&logo=artstation&logoColor=white&color=%2314B8A6)](https://vladmandic.github.io/sdnext/)
[![Code](https://img.shields.io/badge/Code-blue?style=for-the-badge&logo=github&logoColor=white&color=%232563EB)](https://github.com/vladmandic/sdnext)
[![Docs](https://img.shields.io/badge/Docs-purple?style=for-the-badge&logo=gitbook&logoColor=white&color=%237C3AED)](https://vladmandic.github.io/sdnext-docs/)
[![Wiki](https://img.shields.io/badge/Wiki-purple?style=for-the-badge&logo=wearos&logoColor=white&color=%23A78BFA)](https://github.com/vladmandic/sdnext/wiki)
[![Changelog](https://img.shields.io/badge/Changelog-purple?style=for-the-badge&logo=git&logoColor=white&color=%23C4B5FD)](CHANGELOG.md)

[![Discord](https://img.shields.io/discord/1101998836328697867?style=for-the-badge&logo=Discord&logoColor=white&color=%233B82F6&svg=true)](https://discord.gg/VjvR2tabEX)
[![Sponsors](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23FB7185&style=for-the-badge)](https://github.com/sponsors/vladmandic)

</div>
<br>

## Table of contents

- [Documentation](https://vladmandic.github.io/sdnext-docs/)
- [Features & Capabilities](#features-and-capabilities)
- [Screenshots](#screenshots)
- [Supported Workflows](#supported-workflows)
- [Supported AI Models](#supported-ai-models)
- [Supported Platforms & Hardware](#supported-platforms-and-hardware)
- [Getting started](#getting-started)
- [Community and Support](#community-and-support)
- [Credits](#credits)
- [Development and Contributing](#development-and-contributing)

### Why SD.Next?

SD.Next is feature-rich open-source AI art generation platform with a focus on performance, flexibility, and user experience.

In addition to supporting all popular [workflows](#supported-workflows), a wide range of [platforms](#supported-platforms-and-hardware) and [models](#supported-ai-models), SD.Next includes many features not found in other WebUIs, such as:

- **SDNQ**: State-of-the-Art model quantization engine
  Use pre-quantized or run with quantization on-the-fly for up to 4x VRAM reduction with no or minimal quality and performance impact  
- **Balanced Offload**: Dynamically balance CPU and GPU memory to run larger models on limited hardware
- **Captioning** with 150+ **OpenCLiP** models, **Tagger** with **WaifuDiffusion** and **DeepDanbooru** models, and 25+ built-in **VLMs**  
- **Image Processing** with full image correction color-grading suite of tools  
- [Multi-platform](https://vladmandic.github.io/sdnext-docs/nVidia/)!  
- Support for many [Diffusion models](https://vladmandic.github.io/sdnext-docs/Model-Support/)!  
- Fully **localized** to ~15 languages and with support for many [UI themes](https://vladmandic.github.io/sdnext-docs/Themes/)!
- [Desktop](#screenshot-desktop-interface) and [Mobile](#screenshot-mobile-interface) support!  
- Platform specific auto-detection and tuning performed on install  
- Built in installer with automatic updates and dependency management  

## Screenshots

<div align="right">Desktop interface</div>

<div align="center">
<img src="https://github.com/vladmandic/sdnext/raw/dev/ui/assets/screenshot-robot.jpg" alt="SD.Next: AI art generator desktop interface screenshot" width="90%">
</div>

<div align="right">Mobile interface</div>

<div align="center">
<img src="https://github.com/user-attachments/assets/ced9fe0c-d2c2-46d1-94a7-8f9f2307ce38" alt="SD.Next: AI art generator mobile interface screenshot" width="35%">
</div>

<br>

<br>

## Supported Workflows

- Generate with *Text-to-Image*, *Image-to-Image*, *Text-to-Video*, *Image-to-Video*, etc.
- Edit with *Image-Edit*, *Inpainting*, *Outpainting*, etc.
- Enhance guidance with *LoRA*, *ControlNet* and *IPAdapters*, etc.
- Process with *Caption*, *Tag*, *Upscale*, *Interpolate*, *Colorize*, *Filter*, etc.
- and many more with support for custom scripts and extensions

## Supported AI Models

SD.Next supports broad range of models and its frequently updated with latest models  
For full list, see [supported models](https://vladmandic.github.io/sdnext-docs/Model-Support/) and [model specs](https://vladmandic.github.io/sdnext-docs/Models/)  

## Supported Platforms and Hardware

SD.Next is designed to run on a wide range of hardware and platforms, with optimizations for various GPU architectures with acceleration and support for CPU-only execution. Supported platforms include:

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

## Community and Support

If you're unsure how to use a feature, best place to start is [Docs](https://vladmandic.github.io/sdnext-docs/) and if its not there,  
check [ChangeLog](https://vladmandic.github.io/sdnext-docs/CHANGELOG/) for when feature was first introduced as it will always have a short note on how to use it  

And for any question, reach out on [Discord](https://discord.gg/VjvR2tabEX) or open an [issue](https://github.com/vladmandic/sdnext/issues) or [discussion](https://github.com/vladmandic/sdnext/discussions)  

### Credits

Main credit goes to [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for the original codebase  

### Development and Contributing

Please see [Dev Home](https://vladmandic.github.io/sdnext-docs/Dev-Home/) for details on how to contribute to this project  
