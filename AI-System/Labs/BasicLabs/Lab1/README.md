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
|          神经网络数据流图           | ![Lab1-dataflow](C:\Users\jsjhf\Desktop\course\coursework\AI-System\imgs\Lab1-dataflow.PNG) |
|损失和正确率趋势图|![Lab1-accuracy](C:\Users\jsjhf\Desktop\course\coursework\AI-System\imgs\Lab1-accuracy.PNG) ![Lab1-loss](C:\Users\jsjhf\Desktop\course\coursework\AI-System\imgs\Lab1-loss.PNG)|
|>网络分析，使用率前十名的操作&nbsp;|---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  --<br/>---------------------------------<br/>Name                         Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     CUDA total %     CUDA total       CUDA time avg    Number of Calls  I<br/>nput Shapes<br/>---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  --<br/>---------------------------------<br/>select                       0.00%            13.300us         0.00%            13.300us         13.300us         0.00%            2.048us          2.048us          1                []<br/><br/>reshape                      0.00%            48.700us         0.00%            48.700us         48.700us         0.00%            8.192us          8.192us          1                []<br/><br/>view                         0.00%            10.700us         0.00%            10.700us         10.700us         0.00%            3.552us          3.552us          1                []<br/><br/>conv2d                       23.30%           1.074s           23.30%           1.074s           1.074s           23.23%           1.074s           1.074s           1                []<br/><br/>convolution                  23.30%           1.074s           23.30%           1.074s           1.074s           23.23%           1.074s           1.074s           1                []<br/><br/>_convolution                 23.29%           1.074s           23.29%           1.074s           1.074s           23.23%           1.074s           1.074s           1                []<br/><br/>contiguous                   0.00%            19.400us         0.00%            19.400us         19.400us         0.00%            71.712us         71.712us         1                []<br/><br/>cudnn_convolution            23.29%           1.074s           23.29%           1.074s           1.074s           23.23%           1.074s           1.074s           1                []<br/><br/>to                           0.00%            7.200us          0.00%            7.200us          7.200us          0.00%            2.048us          2.048us          1                []<br/><br/>detach_                      0.00%            3.400us          0.00%            3.400us          3.400us          0.00%            2.560us          2.560us          1                []<br/><br/>set_                         0.00%            11.700us         0.00%            11.700us         11.700us         0.00%            3.680us          3.680us          1                []<br/><br/>to                           0.00%            15.400us         0.00%            15.400us         15.400us         0.00%            3.746us          3.746us          1                []<br/><br/>detach_                      0.00%            4.100us          0.00%            4.100us          4.100us          0.00%            2.051us          2.051us          1                []<br/><br/>set_                         0.00%            9.300us          0.00%            9.300us          9.300us          0.00%            2.594us          2.594us          1                []<br/><br/>pin_memory                   0.36%            16.761ms         0.36%            16.761ms         16.761ms         0.36%            16.783ms         16.783ms         1                []<br/><br/>is_pinned                    0.00%            42.400us         0.00%            42.400us         42.400us         0.00%            100.348us        100.348us        1                []<br/><br/>empty                        0.00%            6.100us          0.00%            6.100us          6.100us          0.00%            2.496us          2.496us          1                []<br/><br/>set_                         0.00%            3.900us          0.00%            3.900us          3.900us          0.00%            3.266us          3.266us          1                []<br/><br/>pin_memory                   0.00%            77.800us         0.00%            77.800us         77.800us         0.04%            1.957ms          1.957ms          1                []<br/><br/>is_pinned                    0.00%            35.700us         0.00%            35.700us         35.700us         0.04%            1.942ms          1.942ms          1                []<br/><br/>empty                        0.00%            8.000us          0.00%            8.000us          8.000us          0.00%            3.551us          3.551us          1                []<br/><br/>set_                         0.00%            6.000us          0.00%            6.000us          6.000us          0.00%            2.047us          2.047us          1                []<br/><br/>to                           0.00%            8.500us          0.00%            8.500us          8.500us          0.00%            3.266us          3.266us          1                []<br/><br/>detach_                      0.00%            3.300us          0.00%            3.300us          3.300us          0.00%            2.340us          2.340us          1                []<br/><br/>set_                         0.92%            42.256ms         0.92%            42.256ms         42.256ms         0.92%            42.408ms         42.408ms         1                []<br/><br/>to                           0.00%            4.100us          0.00%            4.100us          4.100us          0.00%            2.523us          2.523us          1                []<br/><br/>detach_                      0.00%            3.600us          0.00%            3.600us          3.600us          0.00%            3.172us          3.172us          1                []<br/><br/>set_                         0.00%            6.200us          0.00%            6.200us          6.200us          0.00%            2.531us          2.531us          1                []<br/><br/>pin_memory                   0.00%            183.300us        0.00%            183.300us        183.300us        0.00%            157.695us        157.695us        1                []<br/><br/>is_pinned                    0.00%            35.700us         0.00%            35.700us         35.700us         0.00%            25.023us         25.023us         1                []<br/><br/>empty                        0.00%            4.700us          0.00%            4.700us          4.700us          0.00%            3.398us          3.398us          1                []<br/><br/>set_                         0.00%            3.000us          0.00%            3.000us          3.000us          0.00%            2.211us          2.211us          1                []<br/><br/>pin_memory                   0.00%            62.900us         0.00%            62.900us         62.900us         0.07%            3.288ms          3.288ms          1                []<br/><br/>is_pinned                    0.00%            20.000us         0.00%            20.000us         20.000us         0.00%            11.969us         11.969us         1                []<br/><br/>empty                        0.00%            3.200us          0.00%            3.200us          3.200us          0.00%            2.047us          2.047us          1                []<br/><br/>set_                         0.00%            2.400us          0.00%            2.400us          2.400us          0.00%            2.594us          2.594us          1                []<br/><br/>reshape                      0.00%            21.400us         0.00%            21.400us         21.400us         0.00%            8.125us          8.125us          1                []<br/><br/>view                         0.00%            16.500us         0.00%            16.500us         16.500us         0.00%            2.625us          2.625us          1                []<br/><br/>add                          0.00%            90.700us         0.00%            90.700us         90.700us         0.00%            93.250us         93.250us         1                []<br/><br/>relu                         0.00%            81.300us         0.00%            81.300us         81.300us         0.00%            94.000us         94.000us         1                []<br/><br/>conv2d                       0.01%            242.600us        0.01%            242.600us        242.600us        0.01%            374.500us        374.500us        1                []<br/><br/>convolution                  0.00%            226.400us        0.00%            226.400us        226.400us        0.01%            352.250us        352.250us        1                []<br/><br/>_convolution                 0.00%            210.400us        0.00%            210.400us        210.400us        0.01%            284.750us        284.750us        1                []<br/><br/>contiguous                   0.00%            16.100us         0.00%            16.100us         16.100us         0.00%            63.625us         63.625us         1                []<br/><br/>cudnn_convolution            0.00%            108.000us        0.00%            108.000us        108.000us        0.00%            90.750us         90.750us         1                []<br/><br/>reshape                      0.00%            10.100us         0.00%            10.100us         10.100us         0.00%            7.875us          7.875us          1                []<br/><br/>view                         0.00%            6.900us          0.00%            6.900us          6.900us          0.00%            2.000us          2.000us          1                []<br/><br/>add                          0.00%            44.700us         0.00%            44.700us         44.700us         0.00%            90.625us         90.625us         1                []<br/><br/>relu                         0.00%            51.500us         0.00%            51.500us         51.500us         0.00%            75.750us         75.750us         1                []<br/><br/>max_pool2d                   0.00%            132.100us        0.00%            132.100us        132.100us        0.00%            105.000us        105.000us        1                []<br/><br/>max_pool2d_with_indices      0.00%            102.100us        0.00%            102.100us        102.100us        0.00%            67.625us         67.625us         1                []<br/><br/>feature_dropout              0.00%            220.000us        0.00%            220.000us        220.000us        0.00%            172.000us        172.000us        1                []<br/><br/>empty                        0.00%            20.100us         0.00%            20.100us         20.100us         0.00%            11.250us         11.250us         1                []<br/><br/>bernoulli_                   0.00%            30.600us         0.00%            30.600us         30.600us         0.00%            39.000us         39.000us         1                []<br/><br/>div_                         0.00%            54.700us         0.00%            54.700us         54.700us         0.00%            19.250us         19.250us         1                []<br/><br/>mul                          0.00%            70.500us         0.00%            70.500us         70.500us         0.00%            55.000us         55.000us         1                []<br/><br/>flatten                      0.00%            35.300us         0.00%            35.300us         35.300us         0.00%            139.625us        139.625us        1                []<br/><br/>reshape                      0.00%            10.700us         0.00%            10.700us         10.700us         0.00%            7.750us          7.750us          1                []<br/><br/>view                         0.00%            6.900us          0.00%            6.900us          6.900us          0.00%            2.000us          2.000us          1                []<br/><br/>t                            0.00%            23.100us         0.00%            23.100us         23.100us         0.00%            1.875us          1.875us          1                []<br/><br/>addmm                        5.47%            252.421ms        5.47%            252.421ms        252.421ms        5.46%            252.509ms        252.509ms        1                []<br/><br/>relu                         0.00%            161.700us        0.00%            161.700us        161.700us        0.00%            126.875us        126.875us        1                []<br/><br/>feature_dropout              0.00%            163.000us        0.00%            163.000us        163.000us        0.01%            273.000us        273.000us        1                []<br/><br/>empty                        0.00%            20.500us         0.00%            20.500us         20.500us         0.00%            14.125us         14.125us         1                []<br/><br/>bernoulli_                   0.00%            23.900us         0.00%            23.900us         23.900us         0.00%            57.125us         57.125us         1                []<br/><br/>div_                         0.00%            39.500us         0.00%            39.500us         39.500us         0.00%            22.250us         22.250us         1                []<br/><br/>mul                          0.00%            52.600us         0.00%            52.600us         52.600us         0.00%            98.375us         98.375us         1                []<br/><br/>t                            0.00%            11.700us         0.00%            11.700us         11.700us         0.00%            3.000us          3.000us          1                []<br/><br/>addmm                        0.00%            79.100us         0.00%            79.100us         79.100us         0.00%            106.125us        106.125us        1                []<br/><br/>log_softmax                  0.00%            97.800us         0.00%            97.800us         97.800us         0.11%            5.179ms          5.179ms          1                []<br/><br/>_log_softmax                 0.00%            74.100us         0.00%            74.100us         74.100us         0.00%            94.500us         94.500us         1                []<br/><br/>---------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  --<br/>---------------------------------<br/>Self CPU time total: 4.612s<br/>CUDA time total: 4.625s|
|||


2. 网络分析，不同批大小结果比较

|||
|:----:|:------------:|
|批大小 &nbsp;| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 结果比较 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
|1|Test set: Average loss: 0.1222, Accuracy: 9701/10000 (97%)|
|16|Test set: Average loss: 0.0357, Accuracy: 9886/10000 (99%)|
|64|Test set: Average loss: 0.0287, Accuracy: 9907/10000 (99%)|
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


