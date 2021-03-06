## Read our paper:
https://github.com/josiahcoad/PersonReIdentification/blob/master/Improvements_Upon_the_Multiple_Granularities_Networkfor_Person_Reid.pdf

## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- TorchVision
- Matplotlib
- Argparse
- Sklearn
- Pillow
- Numpy
- Scipy
- Tqdm

## Train

### Prepare training data

Download Market1501 training data.[here](http://www.liangzheng.org/Project/project_reid.html)

### Begin to train

In the demo.sh file, add the Market1501 directory to --datadir

run `sh demo.sh`

##  Our Results

|  | mAP | rank1 | rank3 | rank5 | rank10 |
| :------: | :------: | :------: | :------: | :------: | :------: |
| 2018-7-22 | 92.17 | 94.60 | 96.53 | 97.06 | 98.01 |
| 2018-7-24 | 93.53 | 95.34 | 97.06 | 97.68 | 98.49 |
| last | 93.83 | 95.78 | 97.21 | 97.83 | 98.43 |

Download model file in [here](https://pan.baidu.com/s/1DbZsT16yIITTkmjRW1ifWQ)


## The architecture of Multiple Granularity Network (MGN)
![Multiple Granularity Network](https://pic2.zhimg.com/80/v2-90a8763a0b7aa86d9152492eb3f85899_hd.jpg)

Figure . Multiple Granularity Network architecture.
