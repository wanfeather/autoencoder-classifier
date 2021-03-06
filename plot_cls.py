import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('loss.csv')

fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(df.iloc[:, 0], df.iloc[:, 1], color = 'red', label = 'training loss')
ax.set_ylabel('training loss', color = 'black', fontsize = 20)
ax.tick_params(axis = 'y', labelcolor = 'black')
ax.legend(loc = 'upper left')

ax2 = ax.twinx()
ax2.plot(df.iloc[:, 0], df.iloc[:, 2], color = 'green', label = 'training accuracy')
ax2.plot(df.iloc[:, 0], df.iloc[:, 3], color = 'blue', label = 'testing accuracy')
ax2.set_ylabel('accuracy', color = 'black', fontsize = 20)
ax2.tick_params(axis = 'y', labelcolor = 'black')
ax2.legend(loc = 'lower left')

plt.savefig('result_cls.png')
