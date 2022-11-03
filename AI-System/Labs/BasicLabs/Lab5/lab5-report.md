# 实验5报告

## 1. 实验环境

### 1.1 硬件环境

CPU(vCPU数目): `Intel® Core™ i7-9750H CPU @ 2.60GHz × 12 `

GPU(型号，数目): `GeForce RTX 2080 with Max-Q Design/PCIe/SSE2 × 1`

### 1.2 软件环境

OS版本: `Ubuntu 18.04.5 LTS`

深度学习框架, python包名称及版本: `Pytorch 1.5, Tensorflow 1.15.0`

CUDA版本: 11.0

## 2. 实验结果

使用Docker部署PyTorch MNIST 训练程序，以交互的方式在容器中运行训练程序。提交以下内容：

1. 创建模型训练镜像，并提交Dockerfile
2. 提交镜像构建成功的日志
3. 启动训练程序，提交训练成功日志（例如：MNIST训练日志截图）

![build_train_dl](C:\Users\jsjhf\Desktop\course\coursework\AI-System\Labs\BasicLabs\Lab5\img\build_train_dl.png)

![run_train_dl](C:\Users\jsjhf\Desktop\course\coursework\AI-System\Labs\BasicLabs\Lab5\img\run_train_dl.png)

使用Docker部署MNIST模型的推理服务，并进行推理。提交以下内容：

1. 创建模型推理镜像，并提交Dockerfile
2. 启动容器，访问TorchServe API，提交返回结果日志
3. 使用训练好的模型，启动TorchServe，在新的终端中，使用一张手写字体图片进行推理服务。提交手写字体图片，和推理程序返回结果截图。

![run_infer](C:\Users\jsjhf\Desktop\course\coursework\AI-System\Labs\BasicLabs\Lab5\img\run_infer.png)

![kitten](C:\Users\jsjhf\Desktop\course\coursework\AI-System\Labs\BasicLabs\Lab5\img\kitten.jpg)

![infer_get](C:\Users\jsjhf\Desktop\course\coursework\AI-System\Labs\BasicLabs\Lab5\img\infer_get.png)

