import matplotlib.pyplot as plt

x = [2,4,6,8,10]
y = [6,7,8,2,4]

x2 = [1,3,5,7,9]
y2 = [7,8,4,9,1]

plt.bar(x, y, label='Bars 1', color='r')
plt.bar(x2, y2, label='Bars 2', color='g')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nSubtitle?')
plt.legend()
plt.show()