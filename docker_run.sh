#!/bin/bash

# Use provided image name or default to irail:clip
IMAGE=${1:-irail:clip}
shift 2>/dev/null || true

docker run --gpus all \
    -v $(pwd):/opt/project \
    -v /BARO_Cluster/data/data/:/data \
    -v /data1:/model \
    -v /data2:/data2 \
    -v /data3:/data3 \
    -v /home:/home \
    -it \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    ${IMAGE} "$@" 