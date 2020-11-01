## datasets.py

数据处理部分，主要完成对数字信息的读取，解析预处理，对词库进行分词，去标点等操作，最后建立语料库。因为语料库中存在繁体字，所以进行繁->简的变换。

利用结巴分词，和utils.py里面的常用函数。normalize函数和cht_to_chs转换简体。lang（）为对平行语料库中词典的创建。readlangs()为文本解析的类：首先读取文件，然后根据两句话来进行训练。英文直接分词，汉语进行langconv以后分词。



## langconv.py

进行语料库繁体转简体的操作。 

## models.py

利用encoderRNN来进行encoder语料库，DecoderRNN进行decode操作。forword为前向推理计算。AttenDecoderRNN（）基于注意力的解码。主函数为测试神经网络结果。

## trainer.py

tensorsFromPair把词典对转换成一个tensor.listTotensor为输出，采用相同的方法。
encoder_optimizer =optim.SGD(encoder.parameters(), lr=lr)。criterion = nn.NLLLoss()为损失函数。

encoder和decoder各是一个网络，lossfunction计算lebel和预测差作为误差

最后保存结果pth部分需要更改

## eval.py

利用编码器和解码器中进行推理运算。

```
encoder.load_state_dict(torch.load("models/encoder_1000000.pth"))
decoder.load_state_dict(torch.load("models/decoder_1000000.pth"))
```

加载训练好的参数，然后对pairs进行遍历。

```
n_iters = 10
train_sen_pairs = [
random.choice(pairs) for i in range(n_iters)
]
training_pairs = [
tensorsFromPair(train_sen_pairs[i]) 
for i in range(n_iters)
]
#用来测试
```

最后可输出分词和翻译结果