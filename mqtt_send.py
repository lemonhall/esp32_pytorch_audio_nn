import serial
import pandas as pd
import paho.mqtt.client as mqtt
import json

try:
    # 创建串口对象
    ser = serial.Serial('COM5', 115200)
except serial.SerialException as e:
    print(f"Error opening COM5: {e}")

export_csv_filename = 0

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
    else:
        print("Connection failed with code", rc)



client = mqtt.Client()
client.on_connect = on_connect

client.connect("192.168.50.232", 1883, 60)

while True:
    loops_nums = 0
    # 创建一个空的 DataFrame
    df = pd.DataFrame(columns=["slope01","slope12","slope23","slope34","slope45","slope56","slope67"])
    while loops_nums< 120:
        try:
            # 读取数据
            data = ser.readline()
            data_array = []
            data_dict  = {}
            if data:
                decodeLine = data.decode('utf-8').strip()
                #print(decodeLine)
                values = decodeLine.split(',')
                #将合法的数据解析为一个数组
                for value in values:
                    try:
                        #print(value)
                        f_temp = float(value)
                        data_array.append(f_temp)
                    except ValueError:
                        pass
                # 将数组转换为字典
                data_dict = {
                            "slope01": data_array[0], 
                            "slope12": data_array[1], 
                            "slope23": data_array[2],
                            "slope34": data_array[3],
                            "slope45": data_array[4],
                            "slope56": data_array[5],
                            "slope67": data_array[6],
                            }
                #print("打印data_dict啊")
                #print(data_dict)
                # 使用 concat 函数添加
                new_df = pd.DataFrame([data_dict],index=[loops_nums])
                df = pd.concat([df, new_df])
        except Exception as e:
            print(f"programe error: {e}")
            pass
        loops_nums=loops_nums+1
    print("打印df啊")
    print(df)
    # export_csv_filename = export_csv_filename +1
    # df.to_csv('traindata_'+ str(export_csv_filename) +'.csv', index=False)  # 不保存索引
    # 序列化数据框为 JSON 字符串
    serialized_df = json.dumps(df.to_dict())
    client.publish("lemon_ken_mic", serialized_df)

# 关闭串口
ser.close()
print("======================关闭串口====================")

# python -m venv venv
# .\venv\Scripts\Activate.ps1
# pip install pyserial
# pip install paho-mqtt
# pip install pandas
# python mqtt_send.py