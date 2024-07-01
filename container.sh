#!/usr/bin/env bash

cd -- "$(dirname -- "$0")"

if [ "$1" = "--cuda" ] || [ "$1" = "--rocm" ] || [ "$1" = "--cpu" ]; then
    COMPUTE=${1#--}
    echo "COMPUTE: $COMPUTE"
else
    echo "Compute platform not specified, assuming compute platform is CUDA"
    COMPUTE=cuda
fi

docker build -t sd-next -f ./docker/$COMPUTE.Dockerfile .
docker rm "SD-Next"
docker run -it --device /dev/dri -v SD-Next:/workspace -v SD-Next_Venv:/python/venv -v SD-Next_Cache:/root/.cache --group-add video -p 7860:7860 --gpus=all --name "SD-Next" sd-next