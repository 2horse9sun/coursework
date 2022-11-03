# 实验4报告

## 1. 实验环境

### 1.1 硬件环境

CPU(vCPU数目): `Intel® Core™ i7-9750H CPU @ 2.60GHz × 12 `

GPU(型号，数目): `GeForce RTX 2080 with Max-Q Design/PCIe/SSE2 × 1`

### 1.2 软件环境

OS版本: `Ubuntu 18.04.5 LTS`

深度学习框架, python包名称及版本: `Pytorch 1.5, Tensorflow 1.15.0`

CUDA版本: 11.0

## 2. 实验结果

Epoch size: ______14______

|                    |              |                                                              |                                                              |
| ------------------ | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 训练算法           |              | &nbsp; &nbsp; &nbsp; &nbsp; 训练时间 &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; 结果正确率 &nbsp; &nbsp; &nbsp; &nbsp; |
| 串行训练           |              | <center>661.4                                                | <center>99.0                                                 |
| 用Horovod并行      | Device# == 2 | <center>343.6                                                | <center>99.0                                                 |
|                    | Device# == 4 | <center>172.6                                                | <center>99.2                                                 |
| float16(16bit)压缩 | Device# == 2 | <center>305.9                                                | <center>98.7                                                 |
|                    | Device# == 4 | <center>153.1                                                | <center>98.8                                                 |
| float8(8bit)压缩   | Device# == 2 | <center>266.3                                                | <center>98.1                                                 |
|                    | Device# == 4 | <center>133.7                                                | <center>98.0                                                 |
|                    |              |                                                              |                                                              |