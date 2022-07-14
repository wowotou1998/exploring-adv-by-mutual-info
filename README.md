# Exploring Adversarial Examples and Adversarial Robustness of Convolutional Neural Networks by Mutual Information

Official implementation for

- Exploring Adversarial Examples and Adversarial Robustness of Convolutional Neural Networks by Mutual Information. ([arXiv](https://arxiv.org/abs/2207.05756))

For any questions, contact (zhangjiebao2014@mail.ynu.edu.cn).

## Requirements

1. [Python](https://www.python.org/)
2. [Pytorch](https://pytorch.org/)
3. [Torattacks >= 3.2.6](https://github.com/Harry24k/adversarial-attacks-pytorch)
4. [Torchvision](https://pytorch.org/vision/stable/index.html)
5. [Pytorchcv](https://github.com/osmr/imgclsmob)

## Preparations
- some file paths will be created manually


## Estimate the mutual information in normal or adversarial training

```
python MI_flow_in_training.py
```
## Estimate the mutual information while the input suffer from the information distortion

```
python MI_flow_in_forward.py
```


[//]: # (## References)



## Citation

If you find this repo useful for your research, please consider citing the paper

```
@misc{https://doi.org/10.48550/arxiv.2207.05756,
  doi = {10.48550/ARXIV.2207.05756},
  url = {https://arxiv.org/abs/2207.05756},
  author = {Zhang, Jiebao and Qian, Wenhua and Nie, Rencan and Cao, Jinde and Xu, Dan},
  title = {Exploring Adversarial Examples and Adversarial Robustness of Convolutional Neural Networks by Mutual Information},
  publisher = {arXiv},
  year = {2022},
}
```
