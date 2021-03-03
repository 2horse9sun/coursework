# Lab 1 - 框架及工具入门示例

## 实验目的

1. 了解深度学习框架及工作流程（Deep Learning Workload）
2. 了解在不同硬件和批大小（batch_size）条件下，张量运算产生的开销


## 实验环境

* PyTorch==1.5.0

* TensorFlow>=1.15.0

* 【可选环境】 单机Nvidia GPU with CUDA 10.0


## 实验原理

通过在深度学习框架上调试和运行样例程序，观察不同配置下的运行结果，了解深度学习系统的工作流程。

## 实验内容

### 实验流程图

![](/imgs/Lab1-flow.png "Lab1 flow chat")

### 具体步骤

1.	按装依赖包。PyTorch==1.5, TensorFlow>=1.15.0

2.	下载并运行PyTorch仓库中提供的MNIST样例程序。

3.	修改样例代码，保存网络信息，并使用TensorBoard画出神经网络数据流图。

4.	继续修改样例代码，记录并保存训练时正确率和损失值，使用TensorBoard画出损失和正确率趋势图。

5.	添加神经网络分析功能（profiler），并截取使用率前十名的操作。

6.	更改批次大小为1，16，64，再执行分析程序，并比较结果。

7.	【可选实验】改变硬件配置（e.g.: 使用/ 不使用GPU），重新执行分析程序，并比较结果。


## 实验报告

### 实验环境

||||
|--------|--------------|:------------------------:|
|硬件环境|CPU（vCPU数目）|                     12                      |
||GPU(型号，数目)|NVIDA GeForce RTX 2080 with Max-Q Design, 1|
|软件环境|OS版本|10.0.18363 N/A Build 18363|
||深度学习框架<br />python包名称及版本|Pytorch 1.5, Tensorflow 1.15.0|
||CUDA版本|V11.1.74|
||||

### 实验结果

1. 模型可视化结果截图
   
||                                                              |
|:-------------:|:--------------------------|
|          神经网络数据流图           | ![Lab1-dataflow](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-dataflow.PNG) |
|损失和正确率趋势图|![Lab1-accuracy](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-accuracy.PNG) ![Lab1-loss](https://raw.githubusercontent.com/2horse9sun/coursework/main/AI-System/imgs/Lab1-loss.PNG)|
|>网络分析，使用率前十名的操作&nbsp;|---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>Name                         Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     CUDA total %     CUDA total       CUDA time avg    Number of Calls<br/>---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>conv2d                       0.00%            42.100us         80.53%           1.081s           540.312ms        23.31%           1.081s           540.425ms        2<br/>convolution                  0.00%            38.900us         80.53%           1.081s           540.291ms        23.31%           1.081s           540.403ms        2<br/>_convolution                 0.01%            87.000us         80.52%           1.081s           540.272ms        23.31%           1.081s           540.332ms        2<br/>cudnn_convolution            80.50%           1.080s           80.50%           1.080s           540.127ms        23.30%           1.080s           540.127ms        2<br/>addmm                        18.65%           250.224ms        18.65%           250.224ms        125.112ms        5.40%            250.262ms        125.131ms        2<br/>pin_memory                   0.73%            9.774ms          0.75%            10.015ms         2.504ms          1.18%            54.935ms         13.734ms         4<br/>feature_dropout              0.01%            68.500us         0.03%            400.700us        200.350us        0.01%            432.125us        216.062us        2<br/>relu                         0.01%            200.300us        0.01%            200.300us        66.767us         0.01%            251.250us        83.750us         3<br/>is_pinned                    0.01%            198.600us        0.01%            198.600us        49.650us         0.03%            1.214ms          303.425us        4<br/>add                          0.01%            134.200us        0.01%            134.200us        67.100us         0.00%            191.125us        95.562us         2<br/>---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>Self CPU time total: 1.342s<br/>CUDA time total: 4.636s|
|||


2. 网络分析，不同批大小结果比较

|||
|:----:|:-------------|
|批大小 &nbsp;| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 结果比较 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
|1|Test set: Average loss: 0.1222, Accuracy: 9701/10000 (97%)<br />---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>Name                         Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     CUDA total %     CUDA total       CUDA time avg    Number of Calls<br/>---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>conv2d                       0.00%            43.600us         79.96%           1.072s           535.771ms        23.49%           1.072s           535.881ms        2<br/>convolution                  0.00%            39.800us         79.95%           1.071s           535.749ms        23.49%           1.072s           535.859ms        2<br/>_convolution                 0.01%            87.300us         79.95%           1.071s           535.729ms        23.49%           1.072s           535.787ms        2<br/>cudnn_convolution            79.93%           1.071s           79.93%           1.071s           535.584ms        23.48%           1.071s           535.589ms        2<br/>addmm                        19.96%           267.563ms        19.96%           267.563ms        133.782ms        5.87%            267.634ms        133.817ms        2<br/>feature_dropout              0.01%            69.800us         0.03%            390.800us        195.400us        0.01%            489.500us        244.750us        2<br/>relu                         0.02%            204.700us        0.02%            204.700us        68.233us         0.01%            256.625us        85.542us         3<br/>add                          0.01%            135.600us        0.01%            135.600us        67.800us         0.00%            189.000us        94.500us         2<br/>max_pool2d                   0.00%            30.200us         0.01%            131.600us        131.600us        0.00%            104.250us        104.250us        1<br/>mul                          0.01%            131.100us        0.01%            131.100us        65.550us         0.00%            155.625us        77.812us         2<br/>---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>Self CPU time total: 1.340s<br/>CUDA time total: 4.562s|
|16|Test set: Average loss: 0.0357, Accuracy: 9886/10000 (99%)<br />---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>Name                         Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     CUDA total %     CUDA total       CUDA time avg    Number of Calls<br/>---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>conv2d                       0.00%            45.400us         80.03%           1.092s           546.228ms        23.46%           1.093s           546.398ms        2<br/>convolution                  0.00%            38.800us         80.02%           1.092s           546.205ms        23.46%           1.093s           546.323ms        2<br/>_convolution                 0.01%            88.800us         80.02%           1.092s           546.186ms        23.46%           1.093s           546.283ms        2<br/>cudnn_convolution            79.99%           1.092s           79.99%           1.092s           545.996ms        23.45%           1.092s           546.005ms        2<br/>addmm                        18.23%           248.845ms        18.23%           248.845ms        124.422ms        5.35%            248.941ms        124.471ms        2<br/>set_                         1.20%            16.424ms         1.20%            16.424ms         2.053ms          0.34%            15.997ms         2.000ms          8<br/>pin_memory                   0.37%            4.997ms          0.38%            5.194ms          1.299ms          0.16%            7.676ms          1.919ms          4<br/>to                           0.07%            948.200us        0.07%            948.200us        237.050us        0.00%            21.632us         5.408us          4<br/>feature_dropout              0.01%            99.000us         0.04%            518.900us        259.450us        0.01%            554.750us        277.375us        2<br/>relu                         0.03%            342.800us        0.03%            342.800us        114.267us        0.01%            348.625us        116.208us        3<br/>---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>Self CPU time total: 1.365s<br/>CUDA time total: 4.657s|
|64|Test set: Average loss: 0.0287, Accuracy: 9907/10000 (99%)<br />---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>Name                         Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     CUDA total %     CUDA total       CUDA time avg    Number of Calls<br/>---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>conv2d                       0.00%            42.100us         80.53%           1.081s           540.312ms        23.31%           1.081s           540.425ms        2<br/>convolution                  0.00%            38.900us         80.53%           1.081s           540.291ms        23.31%           1.081s           540.403ms        2<br/>_convolution                 0.01%            87.000us         80.52%           1.081s           540.272ms        23.31%           1.081s           540.332ms        2<br/>cudnn_convolution            80.50%           1.080s           80.50%           1.080s           540.127ms        23.30%           1.080s           540.127ms        2<br/>addmm                        18.65%           250.224ms        18.65%           250.224ms        125.112ms        5.40%            250.262ms        125.131ms        2<br/>pin_memory                   0.73%            9.774ms          0.75%            10.015ms         2.504ms          1.18%            54.935ms         13.734ms         4<br/>feature_dropout              0.01%            68.500us         0.03%            400.700us        200.350us        0.01%            432.125us        216.062us        2<br/>relu                         0.01%            200.300us        0.01%            200.300us        66.767us         0.01%            251.250us        83.750us         3<br/>is_pinned                    0.01%            198.600us        0.01%            198.600us        49.650us         0.03%            1.214ms          303.425us        4<br/>add                          0.01%            134.200us        0.01%            134.200us        67.100us         0.00%            191.125us        95.562us         2<br/>---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------<br/>Self CPU time total: 1.342s<br/>CUDA time total: 4.636s|
|||

## 参考代码

1.	MNIST样例程序：

    代码位置：Lab1/mnist_basic.py

    运行命令：`python mnist_basic.py`

2.	可视化模型结构、正确率、损失值

    代码位置：Lab1/mnist_tensorboard.py

    运行命令：`python mnist_tensorboard.py`

3.	网络性能分析

    代码位置：Lab1/mnist_profiler.py

## 参考资料

* 样例代码：[PyTorch-MNIST Code](https://github.com/pytorch/examples/blob/master/mnist/main.py)
* 模型可视化：
  * [PyTorch Tensorboard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) 
  * [PyTorch TensorBoard Doc](https://pytorch.org/docs/stable/tensorboard.html)
  * [pytorch-tensorboard-tutorial-for-a-beginner](https://medium.com/@rktkek456/pytorch-tensorboard-tutorial-for-a-beginner-b037ee66574a)
* Profiler：[how-to-profiling-layer-by-layer-in-pytroch](https://stackoverflow.com/questions/53736966/how-to-profiling-layer-by-layer-in-pytroch)


