import matplotlib.pyplot as plt

population_ages = [22,78,57,47,15,65,47,92,74,115,118,130,128,19,21,102,104,14,9,35,88,74,62,20,105,108]

#ids = [x for x in range(len(population_ages))]

bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nSubtitle?')
plt.legend()
plt.show()