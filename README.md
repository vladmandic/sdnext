<div align="center">
<img src="https://github.com/vladmandic/sdnext/raw/master/html/logo-transparent.png" width=200 alt="SD.Next">

# SD.Next: All-in-one WebUI for AI generative image and video creation and captioning  

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
- [SD.Next Features](#sdnext-features)
- [Model support](#model-support)
- [Platform support](#platform-support)
- [Getting started](#getting-started)

## SD.Next Features

All individual features are not listed here, instead check [ChangeLog](CHANGELOG.md) for full list of changes
- Fully localized:
  ▹ **English | Chinese | Russian | Spanish | German | French | Italian | Portuguese | Japanese | Korean**  
- Desktop and Mobile support!  
- Multiple [diffusion models](https://vladmandic.github.io/sdnext-docs/Model-Support/)!  
- Multi-platform!  
 ▹ **Windows | Linux | MacOS | nVidia CUDA | AMD ROCm | Intel Arc / IPEX XPU | DirectML | OpenVINO | ONNX+Olive | ZLUDA**
- Platform specific auto-detection and tuning performed on install  
- Optimized processing with latest `torch` developments with built-in support for model compile and quantize  
  Compile backends: *Triton | StableFast | DeepCache | OneDiff | TeaCache | etc.*  
  Quantization methods: *SDNQ | BitsAndBytes | Optimum-Quanto | TorchAO / LayerWise*  
- **Captioning** with 150+ **OpenCLiP** models, **Tagger** with **WaifuDiffusion** and **DeepDanbooru** models, and 20+ built-in **VLMs**  
- Built in installer with automatic updates and dependency management  

<br>

**Desktop** interface  
<div align="center">
<img src="https://github.com/user-attachments/assets/d6119a63-6ee5-4597-95f6-29ed0701d3b5" alt="screenshot-modernui-desktop" width="90%">
</div>

**Mobile** interface  
<div align="center">
<img src="https://github.com/user-attachments/assets/ced9fe0c-d2c2-46d1-94a7-8f9f2307ce38" alt="screenshot-modernui-mobile" width="35%">
</div>

For screenshots and information on other available themes, see [Themes](https://vladmandic.github.io/sdnext-docs/Themes/)

<br>

## Model support

SD.Next supports broad range of models: [supported models](https://vladmandic.github.io/sdnext-docs/Model-Support/) and [model specs](https://vladmandic.github.io/sdnext-docs/Models/)  

## Platform support

- *nVidia* GPUs using **CUDA** libraries on both *Windows and Linux*  
- *AMD* GPUs using **ROCm** libraries on *Linux*  
  Support will be extended to *Windows* once AMD releases ROCm for Windows  
- *Intel Arc* GPUs using **OneAPI** with *IPEX XPU* libraries on both *Windows and Linux*  
- Any GPU compatible with *DirectX* on *Windows* using **DirectML** libraries  
  This includes support for AMD GPUs that are not supported by native ROCm libraries  
- Any GPU or device compatible with **OpenVINO** libraries on both *Windows and Linux*  
- *Apple M1/M2* on *OSX* using built-in support in Torch with **MPS** optimizations  
- *ONNX/Olive*  
- *AMD* GPUs on Windows using **ZLUDA** libraries  

Plus Docker container recipes for: [CUDA, ROCm, Intel IPEX and OpenVINO](https://vladmandic.github.io/sdnext-docs/Docker/)

## Getting started

- Get started with **SD.Next** by following the [installation instructions](https://vladmandic.github.io/sdnext-docs/Installation/)  
- For more details, check out [advanced installation](https://vladmandic.github.io/sdnext-docs/Advanced-Install/) guide  
- List and explanation of [command line arguments](https://vladmandic.github.io/sdnext-docs/CLI-Arguments/)  
- Install walkthrough [video](https://www.youtube.com/watch?v=nWTnTyFTuAs)  

> [!TIP]
> And for platform specific information, check out  
> [WSL](https://vladmandic.github.io/sdnext-docs/WSL/) | [Intel Arc](https://vladmandic.github.io/sdnext-docs/Intel-ARC/) | [DirectML](https://vladmandic.github.io/sdnext-docs/DirectML/) | [OpenVINO](https://vladmandic.github.io/sdnext-docs/OpenVINO/) | [ONNX & Olive](https://vladmandic.github.io/sdnext-docs/ONNX-Runtime/) | [ZLUDA](https://vladmandic.github.io/sdnext-docs/ZLUDA/) | [AMD ROCm](https://vladmandic.github.io/sdnext-docs/AMD-ROCm/) | [MacOS](https://vladmandic.github.io/sdnext-docs/MacOS-Python/) | [nVidia](https://vladmandic.github.io/sdnext-docs/nVidia/) | [Docker](https://vladmandic.github.io/sdnext-docs/Docker/)

> [!WARNING]
> If you run into issues, check out [troubleshooting](https://vladmandic.github.io/sdnext-docs/Troubleshooting/) and [debugging](https://vladmandic.github.io/sdnext-docs/Debug/) guides  

### Contributing

Please see [Contributing](CONTRIBUTING) for details on how to contribute to this project  
And for any question, reach out on [Discord](https://discord.gg/VjvR2tabEX) or open an [issue](https://github.com/vladmandic/sdnext/issues) or [discussion](https://github.com/vladmandic/sdnext/discussions)  

### Credits

- Main credit goes to [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for the original codebase  
- Additional credits are listed in [Credits](https://github.com/AUTOMATIC1111/stable-diffusion-webui/#credits)  
- Licenses for modules are listed in [Licenses](html/licenses.html)  

### Evolution

<a href="https://star-history.com/#vladmandic/sdnext&Date">
  <picture width=640>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=vladmandic/sdnext&type=Date&theme=dark" />
    <img src="https://api.star-history.com/svg?repos=vladmandic/sdnext&type=Date" alt="starts" width="320">
  </picture>
</a>

- [OSS Stats](https://ossinsight.io/analyze/vladmandic/sdnext#overview)

### Docs

If you're unsure how to use a feature, best place to start is [Docs](https://vladmandic.github.io/sdnext-docs/) and if its not there,  
check [ChangeLog](https://vladmandic.github.io/sdnext-docs/CHANGELOG/) for when feature was first introduced as it will always have a short note on how to use it  

<br>
