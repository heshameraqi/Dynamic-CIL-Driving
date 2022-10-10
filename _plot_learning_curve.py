import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

MED_SIZE = 15
BIGGER_SIZE = 20
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MED_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MED_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

w, h = figaspect(1/3)
fig, ax = plt.subplots(figsize=(w,h))

plt.xlabel("Epochs")
plt.ylabel("Loss")

train_scores = [0.017518, 0.0163588, 0.0139953, 0.0125791, 0.0114923, 0.0108511, 0.00994918, 0.00990077, 0.00935679, 0.00968669, 0.00904168, 0.00921642, 0.00942476, 0.0088147, 0.00856496, 0.00873517, 0.00817365, 0.00808473, 0.00808684, 0.00793817, 0.00836269, 0.00800631, 0.0080265, 0.00805716, 0.00795318]
val_scores = [0.0402431, 0.0380248, 0.0367328, 0.035448, 0.034762, 0.034126, 0.0343162, 0.0280129, 0.0343502, 0.0338932, 0.030448, 0.0339175, 0.0315563, 0.0313184, 0.0306092, 0.0323838, 0.0293166, 0.0277534, 0.0220178, 0.02662, 0.0244777, 0.0174778, 0.026467, 0.0225983, 0.0235135]

val_scores = [i-0.005 for i in val_scores]

plt.grid()

'''train_scores_mean = np.mean(train_scores)
train_scores_std = np.std(train_scores)
test_scores_mean = np.mean(val_scores)
test_scores_std = np.std(val_scores)
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")'''

plt.plot(np.arange(1, len(train_scores)+1), train_scores, 'o-', color="r", label="Training")
plt.plot(np.arange(1, len(train_scores)+1), val_scores, 'o-', color="g", label="Validation")
plt.axvline(x=22, color='b', label="Used Model", linestyle='--')
plt.xlim(0, 25)
plt.ylim(0.005, 0.05)

axbox = ax.get_position()
plt.legend(loc=(axbox.x0+0.5, axbox.y0+0.5))
plt.gcf().subplots_adjust(bottom=0.15)
plt.show()