"""
@author: Luc Cote
"""
import csv
import matplotlib.pyplot as plt
from ast import literal_eval

Greedy_Trials = ["ADAPTIVEm95_BA_GREEDY.csv", "ADAPTIVEm95_ER_GREEDY.csv",  "ADAPTIVEm95_SBM_GREEDY.csv", "ADAPTIVEm95_WS_GREEDY.csv", "ADAPTIVEm95_CAROAD_GREEDY.csv", "ADAPTIVEm95_INFMAXCalTech_GREEDY.csv", "ADAPTIVEm95_YOUTUBE50_GREEDY.csv"]
PrimalDual_Trials = ["ADAPTIVEm95_BA_Primal-Dual.csv", "ADAPTIVEm95_ER_Primal-Dual.csv", "ADAPTIVEm95_SBM_Primal-Dual.csv", "ADAPTIVEm95_WS_Primal-Dual.csv", "ADAPTIVEm95_CAROAD_Primal-Dual.csv", "ADAPTIVEm95_INFMAXCalTech_Primal-Dual.csv", "ADAPTIVEm95_YOUTUBE50_Primal-Dual.csv"]

Trial_Names = ["BA Graph", "ER Graph", "SBM Graph", "WS Graph", "CA ROADs Dataset", "Caltech Facebook Dataset", "Youtube Reccomendation Dataset"]

def mean(data):
  return sum(data)/len(data)

figure, ax = plt.subplots(2, 4)

dualfit_avgs = []
topk_avgs = []
marginal_avgs = []
curvature_avgs = []

for i in range(len(Greedy_Trials)):
    greedy_vals = []
    dual_vals = []
    dualhist_vals = []
    pd_vals = []
    k_vals = []
    n = 0
    with open("../experiment_results_output_data/"+Greedy_Trials[i]) as greedycsv:
      reader_greedy = csv.reader(greedycsv, delimiter =',')
      next(reader_greedy)
      for row in reader_greedy:
        greedy_vals.append(float(row[0]))
        k_vals.append(int(row[3]))
        n = int(row[4])
    with open("../experiment_results_output_data/"+PrimalDual_Trials[i]) as pdcsv:
       reader_pd = csv.reader(pdcsv, delimiter =',')
       next(reader_pd)
       for row in reader_pd:
           pd_vals.append(float(row[0]))
           dual_vals.append(float(row[3]))
           dualhist_vals.append(literal_eval(row[7]))
    
    ax1 = int(i / 4)
    ax2 = i % 4
    X = k_vals
  
    ax[ax1, ax2].plot(X, pd_vals, color = 'purple', label = 'Primal-Dual', linestyle='-', marker='o')
    ax[ax1, ax2].plot(X, greedy_vals, color = 'r', label = 'Greedy', linestyle='--', marker='o')
    ax[ax1, ax2].set(xlabel='k Value', ylabel='Objective Value')
    ax[ax1, ax2].set_title(Trial_Names[i] + " (n=" +str(n)+")")

figure.delaxes(ax[1, 3])
ax[1][2].legend(loc='center left', bbox_to_anchor=(1.3, 0.5))
plt.show()