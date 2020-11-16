import paddlehub as hub 
'''lac = hub.Module(name='lac')  
test_text =["我爱自然语言处理", "南京市长江大桥", "无线电法国别研究"]
inputs = {"text": test_text} 
results = lac.lexical_analysis(data=inputs) 
for result in results: 
   print(result['word']) 
   print(result['tag']) '''


def ner(zh_str):
    test_text = []
    test_text.append(zh_str)
    lac = hub.Module(name='lac') 
    inputs = {"text": test_text} 
    results = lac.lexical_analysis(data=inputs)
    return {"word": results[0]['word'],
            "tag": results[0]['tag']}

if __name__ == "__main__":
    print(ner("最近几个月，泰国出现由年轻人发起的示威浪潮，呼吁对政府和君主制进行改革。"
              "示威画面与一年前香港反修例风波极为相似。泰国和香港同属“奶茶联盟”，是亚洲青年运动网络的一部份，他们靠什么连结起来？"))
    


