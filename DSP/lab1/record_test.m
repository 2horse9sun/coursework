Record = [];
mkdir ������
fprintf("��β���һ�����飬ÿ���ò�ͬ����ֱ�˵�����Ĵ�0-9ʮ�����֣�ÿ�β�������ʱ��2s��\n���õ����ǵ����Ͽ�\n")
name = input('������������ֵ�����ĸСд(e.g.dxd,hdq,pym)\n','s');
fprintf("���¿ո�ֱ�ӿ�ʼ\n")
pause
for j = 1:3
    fprintf("\n��%d��\n",j)
for i = 1:10
    fprintf(['\n','�ļ�����',name, '_' ,num2str(i-1), '_', num2str(j),'\n'])
    record(2,[name, '_' ,num2str(i-1), '_', num2str(j)],i-1);
    ok = input('¼������,���س������������Ҫ����¼�������룺yes\n','s');
    ok = [ok,' '];
    while ok == 'yes '
        record(2,[name, '_' ,num2str(i-1), '_', num2str(j)],i-1);
        ok = input('¼������,���س������������Ҫ����¼�������룺yes\n','s');
        ok = [ok,' '];
    end
end
end
disp("������¼����ɣ�лл������ϡ������������������������ĸ��д��.\������_xxx�� �������ѧί��")

function [myRecording] = record(time, name, i)
% ¼��¼times����
recObj = audiorecorder(8000, 16, 1);
fprintf('��ʼ¼����˵��%d\n.',i)
recordblocking(recObj, time);
% �ط�¼������
play(recObj);
% ��ȡ¼������
myRecording = getaudiodata(recObj);
% ����¼�����ݲ���
plot(myRecording);
%�洢�����ź�
filename = ['.\������\', name, '.wav']; 
audiowrite(filename,myRecording,8000);
end