import matplotlib.pyplot as plt
import numpy as np

data_path = 'C:/Users/rodri/OneDrive/Documentos/python/data/'

x, y = np.loadtxt(data_path+'example.txt', delimiter=',', unpack=True)

plt.plot(x, y, label='Loaded from file')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nSubtitle?')
plt.legend()
plt.show()