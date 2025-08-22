
import torch
import numpy as np

x = torch.tensor(0.0, requires_grad=True)  # x��Ҫ����
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
optimizer = torch.optim.SGD(params=[x], lr=0.01)  #SGDΪ����ݶ��½�
print(optimizer)
 
def f(x):
    result = a * torch.pow(x, 2) + b * x + c
    return (result)
 
for i in range(500):
    optimizer.zero_grad()  #��ģ�͵Ĳ�����ʼ��Ϊ0
    y = f(x)
    y.backward()  #���򴫲������ݶ�
    optimizer.step()  #�������еĲ���
print("y=", y.data, ";", "x=", x.data)



x=torch.zeros(3,3,dtype=torch.float32,requires_grad=True)
y=x-4
z=y**4*6
out=9*z.mean()
out.backward()

print(x.grad)
