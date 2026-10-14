<div align="center">
<img src="https://github.com/vladmandic/sdnext/raw/dev/ui/assets/logo-transparent.png" width=200 alt="SD.Next: AI art generator logo">

# SD.Next

SD.Next is a state-of-the-art open-source web interface for generating and editing images and video with a wide range of diffusion models. Run it on a variety of NVIDIA, AMD, Intel, Apple Silicon, or CPU setups.

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

## Quick links

- [Install](https://vladmandic.github.io/sdnext-docs/Installation/)
- [Documentation](https://vladmandic.github.io/sdnext-docs/)
- [Troubleshooting](https://vladmandic.github.io/sdnext-docs/Troubleshooting/)
- [Supported Workflows](#supported-workflows)
- [Supported AI Models](#supported-ai-models)
- [Supported Platforms & Hardware](#supported-platforms-and-hardware)
- [Getting started](#getting-started)
- [Community](#community-and-support)
- [Contributing](#development-and-contributing)

## What you can do

- **Generate and edit:** Create **images** or **videos** from text or images; *edit, inpaint, outpaint, refine, detail*
- **Guide generation:** Use LoRA, ControlNet, IP-Adapters, and Prompt enhancement
- **Enhance and process:** Caption, Tag, Upscale, Interpolate, Colorize, and adjust images
- **Work with models:** Browse [supported models](https://vladmandic.github.io/sdnext-docs/Model-Support/), auto-download from the reference list, or add your own; added models are detected automatically
- **Tune for your hardware:** Use [SDNQ quantization and attention](https://github.com/Disty0/sdnq) and CPU/GPU offloading to manage memory
- **Make it yours:** Choose from multiple [themes](https://vladmandic.github.io/sdnext-docs/Themes/) and interface [languages](https://vladmandic.github.io/sdnext-docs/Dev-Locale), on **desktop** or **mobile**
- **Stay current:** The built-in installer manages dependencies and updates

See [supported workflows](#supported-workflows) and [platform guides](https://vladmandic.github.io/sdnext-docs/Platforms/) for more details.

## Screenshots & Video

[![SD.Next Screenshot Video](https://github.com/user-attachments/assets/97536dca-3d0d-4fcd-80ba-7cdd4dad47b1)](https://github.com/user-attachments/assets/3cdf8066-e708-40e1-8dae-79de88b8022b)
[![SD.Next Screenshot Blank](https://github.com/user-attachments/assets/bb70177a-6a09-43e7-8de6-b8d4ae8c8f46)](https://github.com/user-attachments/assets/bb70177a-6a09-43e7-8de6-b8d4ae8c8f46)
[![SD.Next Screenshot Edit](https://github.com/user-attachments/assets/ed1e37cb-3e0b-43f2-9c24-79cc0c36542f)](https://github.com/user-attachments/assets/ed1e37cb-3e0b-43f2-9c24-79cc0c36542f)
[![SD.Next Screenshot Kanvas](https://github.com/user-attachments/assets/8eaa9c25-96bd-4569-8b9c-63ffdddecb1b)](https://github.com/user-attachments/assets/8eaa9c25-96bd-4569-8b9c-63ffdddecb1b)
[![SD.Next Screenshot Gallery](https://github.com/user-attachments/assets/404493ca-069f-45db-bd9a-f35533965ca1)](https://github.com/user-attachments/assets/404493ca-069f-45db-bd9a-f35533965ca1)
[![SD.Next Screenshot DLSS](https://github.com/user-attachments/assets/b9f0a103-7099-4ae3-9928-ea3ae1204325)](https://github.com/user-attachments/assets/b9f0a103-7099-4ae3-9928-ea3ae1204325)
[![SD.Next Screenshot Scheduler](https://github.com/user-attachments/assets/70721d26-aea5-4128-ba0a-046095108e70)](https://github.com/user-attachments/assets/70721d26-aea5-4128-ba0a-046095108e70)

<br>

## Supported Workflows

- **Generate:** Text-to-image, image-to-image, text-to-video, and image-to-video.
- **Edit:** Image editing, inpainting, outpainting, detailer, and hires/refine.
- **Guide:** LoRA, ControlNet, IP-Adapters, and prompt enhancement.
- **Process:** Captioning, tagging, upscaling, interpolation, colorization, and filters.
- **Queue:** Schedule tasks or add them to the queue.
- Extend workflows with custom scripts and extensions.

## Supported AI Models

SD.Next supports a broad and regularly updated range of models. See the [supported models](https://vladmandic.github.io/sdnext-docs/Model-Support/) and [model specifications](https://vladmandic.github.io/sdnext-docs/Models/).

## Supported Platforms and Hardware

SD.Next supports a range of hardware, with GPU acceleration and CPU-only execution. Supported platforms include:

- *nVidia* GPUs using **CUDA** libraries on both *Windows and Linux*  
- *AMD* GPUs using **ROCm** libraries on both *Linux and Windows*
- *AMD* GPUs on Windows using **ZLUDA** libraries 
- *Intel Arc* GPUs using **OneAPI** with *IPEX XPU* libraries on both *Windows and Linux*  
- Any *CPU/GPU* or device compatible with **OpenVINO** libraries on both *Windows and Linux*  
- *Apple M1/M2* on *OSX* using built-in support in Torch with **MPS** optimizations  

Plus **Docker** container recipes for: [CUDA, ROCm, Intel IPEX and OpenVINO](https://vladmandic.github.io/sdnext-docs/Docker/)

## Getting started

Follow the [installation guide](https://vladmandic.github.io/sdnext-docs/Installation/) for prerequisites and platform-specific setup. For advanced options, see [advanced installation](https://vladmandic.github.io/sdnext-docs/Advanced-Install/) and [command-line arguments](https://vladmandic.github.io/sdnext-docs/CLI-Arguments/). There is also an [installation walkthrough](https://www.youtube.com/watch?v=nWTnTyFTuAs).

> [!TIP]
> For platform-specific instructions, see:  
> [nVidia CUDA](https://vladmandic.github.io/sdnext-docs/nVidia/) | [AMD ROCm](https://vladmandic.github.io/sdnext-docs/AMD-ROCm/) | [Intel Arc XPU](https://vladmandic.github.io/sdnext-docs/Intel-ARC/) | [OpenVINO](https://vladmandic.github.io/sdnext-docs/OpenVINO/) | [ZLUDA](https://vladmandic.github.io/sdnext-docs/ZLUDA/) | [MacOS](https://vladmandic.github.io/sdnext-docs/MacOS-Python/) | [WSL](https://vladmandic.github.io/sdnext-docs/WSL/) | [Docker](https://vladmandic.github.io/sdnext-docs/Docker/)

### Quick start

> [!NOTE]
> First launch may take time while dependencies are installed  
> If you run into issues, see [troubleshooting](https://vladmandic.github.io/sdnext-docs/Troubleshooting/) and [debugging](https://vladmandic.github.io/sdnext-docs/Debug/).

**Linux or macOS**

```bash
git clone https://github.com/vladmandic/sdnext
cd sdnext
./webui.sh
```

**Windows Command Prompt**

```bat
git clone https://github.com/vladmandic/sdnext
cd sdnext
webui.bat (for Windows Command Prompt)
webui.ps1 (for Windows PowerShell)
```

**Windows All-in-One UI**

Download the latest `exe` from [SD.Next Launcher Releases](https://github.com/vladmandic/sdnext-launcher/releases) - And run it!  
> [!TIP]
> The All-in-One UI includes all necessary tools, so you don't need to install anything separately.

## Community and Support

Start with the [documentation](https://vladmandic.github.io/sdnext-docs/). For feature history and release notes, see the [changelog](https://vladmandic.github.io/sdnext-docs/CHANGELOG/). Ask questions on [Discord](https://discord.gg/VjvR2tabEX) or [GitHub Discussions](https://github.com/vladmandic/sdnext/discussions); report bugs through [GitHub Issues](https://github.com/vladmandic/sdnext/issues).

### Credits

SD.Next builds on the original codebase from [AUTOMATIC1111's Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

### Development and Contributing

See [Dev Home](https://vladmandic.github.io/sdnext-docs/Dev-Home/) for development and contribution guidelines.
