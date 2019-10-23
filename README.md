# Research-on-Pose-Estimation-of-Objects-1
物体位姿估计研究-1：复现位姿估计论文《Estimating 6D Pose From Localizing Designated Surface Keypoints》

## Pipeline

### Prepare Data

Go to folder [data/](./data) and follow its instructions. After this step, data will be prepared for training.

### Keypoints Designation

Go to folder [gendata/](./gendata) and follow its instructions. After this step, you'll get models and keypoints raw data (in .ply format).

### Object Detector

Go to folder [detect/](./detect) and follow its instructions. After this step, you'll get YOLOv3 pre-trained weights for LINEMOD.

### Keypoint Localization

Go to folder [keypoint/](./keypoint) and follow its instructions. After this step, you'll get SPPE pre-trained weights for LINEMOD.

### Pose Estimation

Run [scripts/pose.sh](./scripts/pose.sh) to estimate the pose. Results will be saved to [results/](results/).

### Evaluation

Run [scripts/eval.sh](scripts/eval.sh) to estimate the pose and evaluate the result

## Results

ADD (-S) accuracy listed below. Eggbox and glue is calculated with ADD(-S).

| Sequence       | 17 SIFT | 17 Cluster | 17 Cluster w/o dpg | 9 Corner | 9 SIFT |
| -------------- | ------- | ---------- | ------------------ | -------- | ------ |
| 01 Ape         | 0.487   | 0.649      | 0.203              | 0.119    | 0.446  |
| 02 Benchvise   | 0.982   | 0.986      | 0.601              | 0.179    | 0.858  |
| 04 Camera      | 0.687   | 0.888      | 0.739              | 0.538    | 0.778  |
| 05 Can         | 0.821   | 0.921      | 0.873              | 0.231    | 0.813  |
| 06 Cat         | 0.645   | 0.863      | 0.661              | 0.318    | 0.632  |
| 08 Driller     | 0.803   | 0.961      | 0.936              | 0.217    | 0.596  |
| 09 Duck        | 0.508   | 0.587      | 0.523              | 0.100    | 0.329  |
| 10 Eggbox      |         | 0.977      | 0.924              | 0.971    | 0.305  |
| 11 Glue        |         | 0.954      | 0.519              | 0.667    | 0.810  |
| 12 Holepuncher | 0.648   | 0.765      | 0.756              | 0.690    | 0.461  |
| 13 Iron        | 0.918   | 0.969      | 0.859              | 0.460    | 0.838  |
| 14 Lamp        |         | 0.973      | 0.904              | 0.675    | 0.961  |
| 15 Phone       | 0.730   | 0.865      | 0.823              | 0.510    | 0.786  |
| Average        |         | 0.874      | 0.717              | 0.437    | 0.663  |

### Occlusion LINEMOD

| Sequence       | Mean ADD Acc |
| -------------- | ------------ |
| 01 Ape         | 0.152        |
| 05 Can         | 0.583        |
| 06 Cat         | 0.227        |
| 08 Driller     | 0.692        |
| 09 Duck        | 0.151        |
| 10 Eggbox      | 0.825        |
| 11 Glue        | 0.620        |
| 12 Holepuncher | 0.468        |
| Average        | 0.464        |

# Reference:

《Estimating 6D Pose From Localizing Designated Surface Keypoints》

https://github.com/ecr23xx/kp6d

https://github.com/hz-ants/betapose
