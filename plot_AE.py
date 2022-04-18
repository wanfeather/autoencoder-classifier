import matplotlib.pyplot as plt
import pandas as pd

loss = pd.read_csv('loss.csv', sep = ',', header = None, names = ['x', 'y'])

plt.plot(loss['x'], loss['y'])
plt.savefig('results_AE.png')
