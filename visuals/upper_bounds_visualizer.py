"""
@author: Luc Cote
"""
import csv
import matplotlib.pyplot as plt
from ast import literal_eval

Greedy_Trials = ["ADAPTIVEm95_BA_GREEDY.csv", "ADAPTIVEm95_ER_GREEDY.csv",  "ADAPTIVEm95_SBM_GREEDY.csv", "ADAPTIVEm95_WS_GREEDY.csv", "ADAPTIVEm95_CAROAD_GREEDY.csv", "ADAPTIVEm95_INFMAXCalTech_GREEDY.csv", "ADAPTIVEm95_YOUTUBE50_GREEDY.csv", "ADAPTIVEm95_INFMAXCitation_GREEDY.csv", "ADAPTIVEm95_MOVIECOVERsubset_GREEDY.csv"]
PrimalDual_Trials = ["ADAPTIVEm95_BA_Primal-Dual.csv", "ADAPTIVEm95_ER_Primal-Dual.csv", "ADAPTIVEm95_SBM_Primal-Dual.csv", "ADAPTIVEm95_WS_Primal-Dual.csv", "ADAPTIVEm95_CAROAD_Primal-Dual.csv", "ADAPTIVEm95_INFMAXCalTech_Primal-Dual.csv", "ADAPTIVEm95_YOUTUBE50_Primal-Dual.csv","ADAPTIVEm95_INFMAXCitation_Primal-Dual.csv", "ADAPTIVEm95_MOVIECOVERsubset_Primal-Dual.csv"]
Upper_Trials = ["ADAPTIVEm95_BA_UPPER_BOUNDS.csv", "ADAPTIVEm95_ER_UPPER_BOUNDS.csv", "ADAPTIVEm95_SBM_UPPER_BOUNDS.csv", "ADAPTIVEm95_WS_UPPER_BOUNDS.csv", "ADAPTIVEm95_CAROAD_UPPER_BOUNDS.csv", "ADAPTIVEm95_INFMAXCalTech_UPPER_BOUNDS.csv", "ADAPTIVEm95_YOUTUBE50_UPPER_BOUNDS.csv","ADAPTIVEm95_INFMAXCitation_UPPER_BOUNDS.csv", "ADAPTIVEm95_MOVIECOVERsubset_UPPER_BOUNDS.csv"]

Trial_Names = ["BA Graph", "ER Graph", "SBM Graph", "WS Graph", "CA ROADs Dataset", "Caltech Facebook Dataset", "Youtube Reccomendation Dataset", "Citations Dataset", "MovieLens Dataset"]

def mean(data):
  return sum(data)/len(data)

figure, ax = plt.subplots(3, 3)

for i in range(len(Greedy_Trials)):
    greedy_vals = []
    dual_vals = []
    dualhist_vals = []
    dualfitmin_vals = []
    dualfittau_vals = []
    dualfitlp_vals = []
    BQS_vals = []
    topk_vals = []
    marginal_vals = []
    curvature_vals = []
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
    with open("../experiment_results_output_data/"+Upper_Trials[i]) as pdcsv:
       reader_pd = csv.reader(pdcsv, delimiter =',')
       next(reader_pd)
       for row in reader_pd:
           BQS_vals.append(float(row[0]))
           topk_vals.append(float(row[1]))
           marginal_vals.append(float(row[2]))
           curvature_vals.append(float(row[3]))
           dualfitmin_vals.append(float(row[4]))
           dualfittau_vals.append(float(row[5]))
           dualfitlp_vals.append(float(row[6]))
    
    ax1 = int(i / 3)
    ax2 = i % 3
    X = k_vals
  
    ax[ax1, ax2].plot(X, dual_vals, color = 'purple', label = 'primdual', linestyle='-', marker='o')
    ax[ax1, ax2].plot(X, [min(dualfitmin_vals[i], dualfittau_vals[i]) for i in range(len(dualfitmin_vals))], color = 'orange', label = 'GreeDual1', linestyle='-', marker='x')
    ax[ax1, ax2].plot(X, dualfitlp_vals, color = 'b', label = 'GreeDual2', linestyle='-', marker='x')
    ax[ax1, ax2].plot(X, BQS_vals, color = 'g', label = 'BQS', linestyle='--', marker='^')
    ax[ax1, ax2].plot(X, topk_vals, color = 'r', label = 'topk', linestyle='--', marker='s')
    ax[ax1, ax2].plot(X, marginal_vals, color = 'pink', label = 'marginal', linestyle='--', marker='p')
    ax[ax1, ax2].plot(X, [l / j for l, j in zip(greedy_vals, curvature_vals)], color = 'brown', label = 'curvature', linestyle='--', marker='*')
    ax[ax1, ax2].set(xlabel='k Value', ylabel='Certificate Value')
    ax[ax1, ax2].set_title(Trial_Names[i] + " (n=" +str(n)+")")


handles, labels = ax[0][0].get_legend_handles_labels()
figure.legend(handles,labels,loc='upper center', ncol=6, fontsize=15)
plt.show()
