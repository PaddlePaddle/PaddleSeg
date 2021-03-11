## The Champion Model of Multi-Temporal Urban Development Challenge at NeurIPS2020

### 1. Approach

For the detailed approaches, please refer to the [document](./approach.pdf).

### 2. Dataset

Download data from one of following links:

option 1: The official dataset link: https://spacenet.ai/sn7-challenge/

option 2: The BaiduYun [link](https://pan.baidu.com/s/1WM0IHup5Uau7FZGQf7rzdA), the access code: 17th 

### 3. Deployment Guide
Please provide the exact steps required to build and deploy the code:
- Build docker image:
```
docker build -t <id> .
```
- Run docker:
```
docker run -v <local_data_path>:/data:ro -v <local_writable_area_path>:/wdata -it <id>
```
Please see https://github.com/topcoderinc/marathon-docker-template/tree/master/data-plus-code-style

### 4. Final Verification
Please provide instructions that explain how to train the algorithm and have it execute against sample data:
- Train:
```
./train.sh /data/train
```
- Test:
```
./test.sh /data/test/ solution.csv
```

### 5. Team Members
-  Xiang Long, Honghui Zheng, Yan Peng
