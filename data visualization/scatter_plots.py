import matplotlib.pyplot as plt


x = [1,2,3,4,5,6,7,8]
y = [5,4,7,6,9,8,2,1]

# marker = type (o, *) - there are a lot of options
# s = marker size
plt.scatter(x, y, label='Scatter Plot', color='k', marker='*', s=100)


plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nSubtitle?')
plt.legend()
plt.show()