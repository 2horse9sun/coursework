# 实验报告

https://github.com/2horse9sun/coursework/blob/main/AI-System/Labs/BasicLabs/Lab1/lab1-report.md

（图片存储在github仓库中，若因网络原因无法显示，请查看pdf文件）

## 1. 实验环境

### 1.1 硬件环境

CPU(vCPU数目): `Intel® Core™ i7-9750H CPU @ 2.60GHz × 12 `

GPU(型号，数目): `GeForce RTX 2080 with Max-Q Design/PCIe/SSE2 × 1`

### 1.2 软件环境

OS版本: `Ubuntu 18.04.5 LTS`

深度学习框架, python包名称及版本: `Pytorch 1.5, Tensorflow 1.15.0`

CUDA版本: 11.0

## 2.实验结果

### 2.1 模型可视化结果截图

神经网络数据流图:

![Lab1-dataflow](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-dataflow.PNG)

正确率趋势图:

![Lab1-accuracy](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-accuracy.PNG)

损失趋势图: ![Lab1-loss](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-loss.PNG)

网络分析，使用率前十名的操作(不使用CUDA): 

![Lab1-result-batchsize64](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-result-batchsize64.png)

网络分析，使用率前十名的操作(使用CUDA): 

![Lab1-result-batchsize64-cuda](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-result-batchsize64-cuda.png)

通过分析可以看出，卷积操作占用了大多数CPU时间，因此对卷积进行重点优化可以提升性能。

### 2.2 网络分析，不同批大小结果比较

不使用CUDA:

batch_size= 1:

![Lab1-result-batchsize1](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-result-batchsize1.png)

batch_size= 16:

![Lab1-result-batchsize16](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-result-batchsize16.png)

batch_size= 64:

![Lab1-result-batchsize64](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-result-batchsize64.png)

使用CUDA:

batch_size= 1:

![Lab1-result-batchsize1-cuda](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-result-batchsize1-cuda.png)

batch_size= 16:

![Lab1-result-batchsize16-cuda](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-result-batchsize16-cuda.png)

batch_size= 64:

![Lab1-result-batchsize64-cuda](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-result-batchsize64-cuda.png)
