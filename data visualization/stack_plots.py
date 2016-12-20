import matplotlib.pyplot as plt

days = [1,2,3,4,5]

sleeping = [7,8,6,11,7]
eating = [2,3,4,3,2]
working = [7,8,7,2,2]
playing = [8,5,7,8,13]

# This is the 'only way' to put a legend in stack plot
# [] and [] are just 'empty' list, so we won't pass any value
plt.plot([], [], color='m', label='Sleeping', linewidth=5)
plt.plot([], [], color='c', label='Eating', linewidth=5)
plt.plot([], [], color='k', label='Working', linewidth=5)
plt.plot([], [], color='g', label='Playing', linewidth=5)

plt.stackplot(days, sleeping, eating, working, playing, colors=['m','c','k','g'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nSubtitle?')
plt.legend()
plt.show()