from matplotlib import pyplot as plt
import numpy as np

x_array = []
y_array = []
y2_array = []
j = 0
for i in range(10):
    x = []
    x.append(10 * i + 10 * 10 *j)
    y = np.power(10, i)
    y2 = np.power(2, i)
    x_array.append(x)
    y_array.append(y)
    y2_array.append(y2)


plt.plot(np.reshape(x_array, -1), np.reshape(y_array, -1), np.reshape(x_array, -1), np.reshape(y2_array, -1))
plt.show()
plt('efs')