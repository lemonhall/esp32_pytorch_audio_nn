import torch
import pandas as pd
import paho.mqtt.client as mqtt
import json
import tkinter as tk

# 定义神经网络类
class AudioClassifier(torch.nn.Module):
    def __init__(self, batch_size, in_features):
        super(AudioClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(in_features, 128)
        self.additional_layer = torch.nn.Linear(128, 64)
        self.layer2 = torch.nn.Linear(64, 32)
        self.layer3 = torch.nn.Linear(32, 2)  # 假设两类

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.additional_layer(x))  # 使用新增层
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
model = AudioClassifier(1,119*7)  # 定义模型结构
model.load_state_dict(torch.load('odel_weights.pth'))
model.eval()  # 设置为评估模式

# # 输入数据
# input_data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

# # 使用模型进行预测
# output = model(input_data)

# 对输出进行处理或查看结果
# print(output)

def update_circle(probabilities):
    print("====update_circle我拿到了一个啥:====")
    print(probabilities)  # 输出新张量的形状
    if probabilities[0] > 0.8:
        canvas.itemconfig(circle, fill='red')
    else:
        canvas.itemconfig(circle, fill='lightblue')

        
root = tk.Tk()
canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

circle = canvas.create_oval(100, 100, 300, 300, fill='lightblue')


def on_message(client, userdata, msg):
    received_data = json.loads(msg.payload.decode())
    df = pd.DataFrame(received_data)
    #df.values天然就已经去掉表头了，pandas里面的这个语法就是在取值
    print(df.values)
    values119 = df.values[:-1]
    # 将数据转换为张量
    tensor = torch.tensor(values119)
    print("====reviced tensor shape:====")
    print(tensor.shape)  # 输出新张量的形状
    new_predict_tensor = tensor.reshape(119*7)
    # 使用模型进行预测
    output = model(new_predict_tensor.to(torch.float32))
    #对输出进行处理或查看结果
    print(output)
    print("====output tensor shape:====")
    print(output.shape)  # 输出新张量的形状
    print("==========probabilities:============")
    probabilities = torch.softmax(output, dim=0)
    print(probabilities)
    update_circle(probabilities)

client = mqtt.Client()
client.on_message = on_message

client.connect("192.168.50.232", 1883,65535)
client.subscribe("lemon_ken_mic")
client.loop_start()

root.mainloop()

# python -m venv venv
# .\venv\Scripts\Activate.ps1
# pip install pyserial
# python main.py
# pip install pandas
# pip install PyQt5