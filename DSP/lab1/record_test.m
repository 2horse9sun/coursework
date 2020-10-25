Record = [];
mkdir 语音库
fprintf("这次测试一共三组，每组用不同语调分别说出中文从0-9十个数字，每次采样持续时间2s，\n最后得到我们的语料库\n")
name = input('请输入你的名字的首字母小写(e.g.dxd,hdq,pym)\n','s');
fprintf("按下空格直接开始\n")
pause
for j = 1:3
    fprintf("\n第%d组\n",j)
for i = 1:10
    fprintf(['\n','文件名：',name, '_' ,num2str(i-1), '_', num2str(j),'\n'])
    record(2,[name, '_' ,num2str(i-1), '_', num2str(j)],i-1);
    ok = input('录音结束,按回车继续，如果想要重新录入请输入：yes\n','s');
    ok = [ok,' '];
    while ok == 'yes '
        record(2,[name, '_' ,num2str(i-1), '_', num2str(j)],i-1);
        ok = input('录音结束,按回车继续，如果想要重新录入请输入：yes\n','s');
        ok = [ok,' '];
    end
end
end
disp("语音库录入完成，谢谢您的配合。请把语音库后面加上名字首字母缩写‘.\语音库_xxx’ 打包发给学委。")

function [myRecording] = record(time, name, i)
% 录音录times秒钟
recObj = audiorecorder(8000, 16, 1);
fprintf('开始录音请说：%d\n.',i)
recordblocking(recObj, time);
% 回放录音数据
play(recObj);
% 获取录音数据
myRecording = getaudiodata(recObj);
% 绘制录音数据波形
plot(myRecording);
%存储语音信号
filename = ['.\语音库\', name, '.wav']; 
audiowrite(filename,myRecording,8000);
end