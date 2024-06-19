import pandas as pd
import numpy as npy
import matplotlib.pyplot as plt

from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource

plt.rc("font", family='DengXian')  # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 负号报错

data = pd.read_csv(open('traindata_c0_3.csv', encoding='utf-8'))
# 读取数据
print(data)

# 我将要生成一个从0到6，一共7个值的X坐标，分别代表了7个斜率值
# np.arange(0, 7, 1) 会生成一个从 0 （包含）到 7 （不包含），步长为 1 的 numpy 数组。
# [0 1 2 3 4 5 6]
x = npy.arange(0, 7, 1)
# 我将要生成一个从0到120，一共120个值的Y坐标，分别代表了我拥有120个采样向量
#[  0   1   2   3  ..... 115 116 117 118 119]
y = npy.arange(0, 120, 1)
#np.meshgrid 函数用于从给定的一维数组中生成二维的网格坐标矩阵。在上述示例中，xx 和 yy 都是二维数组，分别表示 x 和 y 方向的坐标。
xx, yy = npy.meshgrid(x, y)

# [[0 1 2 3 4 5 6]
#  [0 1 2 3 4 5 6]
# .................
#  [0 1 2 3 4 5 6]
#  [0 1 2 3 4 5 6]]
# 现在的xx长这样，是一个二维数组

# [[  0   0   0   0   0   0   0]
#  [  1   1   1   1   1   1   1]
# ...............................
#  [118 118 118 118 118 118 118]
#  [119 119 119 119 119 119 119]]
# 现在的yy长这样，也是一个二维数组

# 假设数据中的列名与您的 x 坐标相对应
z_matrix = data.values  # 将数据框转换为 numpy 数组

zz = z_matrix[yy, xx]

# 绘制曲面
# plot_surface(X,Y,Z)中X,Y,Z必须是二维数组
f = plt.figure()
plt.axis('off')

fig = f.add_subplot(projection='3d')
# 这段代码通常是在使用 matplotlib 库进行三维绘图时出现的。
# 它使用 plot_surface 函数绘制一个三维曲面。
# xx、yy 和 zz 分别是曲面在 x、y 和 z 方向上的坐标数据。
# alpha=0.7 表示曲面的透明度为 0.7。
# rstride=10 和 cstride=10 用于控制曲面网格的稀疏程度，减少绘制的点数，从而提高绘制速度。
# cmap=plt.cm.coolwarm 则指定了颜色映射，使用 coolwarm 颜色方案来为曲面着色。
surf = fig.plot_surface(xx, yy, zz, alpha=1,
                        rstride=1, cstride=1,
                        cmap=plt.cm.coolwarm)
fig.set_zlabel('Z')
fig.set_ylabel('Y')
fig.set_xlabel('X')

# aspect 色条的长宽比， shrink 色条整体大小缩放
f.colorbar(surf, aspect=6, shrink=0.6)
plt.show()
# exit()