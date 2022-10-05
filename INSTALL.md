## Environment

* We developed SGT with CUDA==10.1 & python3.6 & torch==1.7.1 & detectron2@v0.4.1 & torch-geometric==1.6.3 [dockerfile](docker/sgt_cu101/Dockerfile)
* We also validated SGT with CUDA==11.3 & python3.8 & torch==1.10.2 & detectron2@v0.6 & torch-geometric==2.1.0 [dockerfile](docker/sgt_cu113/Dockerfile)
* We provide Docker Image for your convenience
* Please note that running time was measured using the version of CUDA==10.1

## Start with Docker Image (with docker ≥ 19.03)
```
## Step 1. Download required docker image
# CUDA:10.1 - Note that it is not compatible with the latest GPUs (e.g., RTX30xx)
docker pull jshyunaa/sgt:cu101

# CUDA:11.3
docker pull jshyunaa/sgt:cu113

## Step 2. Create docker container
[Template]
docker run --rm -it -d --shm-size=<mem> --gpus '"device=<device-indices>"'  -v <host_root_dir>:<container_root_dir>\
              --name <container_name> -p <ssh_port>:22 -p <log_port>:8081 jshyunaa/sgt:<version>
[Example]
docker run --rm -it -d --shm-size=16G --gpus '"device=0,1"' -v /media/ssd1/users/jshyun/:/root --name sgt -p 3000:22 -p 3001:8081 jshyunaa/sgt:cu113

## Step 3. Access to the created container
docker exec -it <container_name> /bin/bash
cd <work_dir>
```

* If you want to use conda, please refer the dockerfile to install the required packages. 
* If you use TensorboardXWriter, please downgrade setuptools==59.5.0 ([issue](https://github.com/pytorch/pytorch/pull/69904)) and 
  comment out TensorboardXWriter in the file of projects/EpochTrainer/epoch_trainer/default_epoch_trainer.py

