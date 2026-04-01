#!/usr/bin/env bash

node cli/api-txt2img.js
node cli/api-pulid.js

source venv/bin/activate
echo image-exif
python cli/api-info.py --input html/logo-bg-0.jpg
echo txt2img
python cli/api-txt2img.py --detailer --prompt "girl on a mountain" --seed 42 --sampler DEIS --width 1280 --height 800 --steps 10
echo img2img
python cli/api-img2img.py --init html/logo-bg-0.jpg --steps 10
echo inpaint
python cli/api-img2img.py --init html/logo-bg-0.jpg --mask html/logo-dark.png --steps 10
echo upscale
python cli/api-upscale.py --input html/logo-bg-0.jpg --upscaler "ESRGAN 4x Valar" --scale 4
echo vqa
python cli/api-vqa.py --input html/logo-bg-0.jpg
echo detailer
python cli/api-detect.py --image html/invoked.jpg
echo faceid
python cli/api-faceid.py --face html/simple-dark.jpg
echo control-txt2img
python cli/api-control.py --prompt "cute robot"
echo control-img2img
python cli/api-control.py --prompt "cute robot" --input html/logo-bg-0.jpg
echo control-ipsadapter
python cli/api-control.py --prompt "cute robot" --ipadapter "Base SDXL:html/logo-bg-0.jpg:0.8"
echo control-preprocess
python cli/api-preprocess.py --input html/logo-bg-0.jpg --model "Zoe Depth"
echo control-controlnet
python cli/api-control.py --prompt "cute robot" --input html/logo-bg-0.jpg --type controlnet --control "Zoe Depth:Xinsir Union XL:0.5"
