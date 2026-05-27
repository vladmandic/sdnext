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
