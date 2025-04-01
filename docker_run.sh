#!/bin/bash

# Use provided image name or default to irail:clip
IMAGE=${1:-irail:clip}
shift 2>/dev/null || true

# Get current user ID and group ID
USER_ID=$(id -u)
USER_NAME=$(whoami)
GROUP_ID=$(id -g)

docker run --gpus all \
    -v $(pwd):/opt/project \
    -v /BARO_Cluster/data/data/:/data \
    -v /data1:/model \
    -v /data2:/data2 \
    -v /data3:/data3 \
    -v /home:/home \
    -v /home/$USER_NAME/.cache:/home/$USER_NAME/.cache \
    -v /home/$USER_NAME/.ssh:/home/$USER_NAME/.ssh \
    -e USER_ID=$USER_ID \
    -e GROUP_ID=$GROUP_ID \
    -e USER_NAME=$USER_NAME \
    --user $USER_ID:$GROUP_ID \
    -it \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    ${IMAGE} "$@" 