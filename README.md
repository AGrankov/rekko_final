# Building docker environment
You are need to setup nvidia docker (version 2.0) for faster and easier executing of code:
[Installation guide](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)/)

After nvidia docker was correctly installed build docker image from rekko_final directory
```
docker build -t rekko2019:gpu ./docker
```
Make volume for using inside docker container, make sure, that you're redefine device parameter to proper path
```
docker volume create --name REKKO2019 --opt type=none --opt device=path/to/rekko_final/ --opt o=bind
```
# Evaluate on pretrained models
Making submission based on previously pretrained models was tested on such computer spec:
  -cpu: i7-7800X
  -gpu: 2xGTX 1080 Ti
  -ram: 64Gb + 128Gb swap
Approximate time for executing is 25 minutes.

Interesting note: For retrieving exactly same result as competition last score submission
evaluation scripts should be runned with environment options (CUDA_VISIBLE_DEVICES) for two and one
GPU (run_evaluate.sh script). torch.nn.DataParallel class object execution result is a slightly different on different amount of GPU's, so in a case of running on only one GPU result would be
a bit different.
```
docker run -v REKKO2019:/REKKO2019 --runtime=nvidia --name REKKO2019 -t rekko2019:gpu bash -c 'cd /REKKO2019/scripts/; bash ./run_evaluate.sh'
```
Result archive was placed under rekko_final/submissions folder
# Traininig and evaluation
Computer spec same as at evaluation section
Approximate time for full execution of training and evaluation process is 2 days
```
docker run -v REKKO2019:/REKKO2019 --runtime=nvidia --name REKKO2019 --shm-size=64g -t rekko2019:gpu bash -c 'cd /REKKO2019/scripts/; bash ./run_training_evaluate.sh'
```
