import csv
import matplotlib.pyplot as plt

Greedy_Trials = ["ADAPTIVEm95_BA_GREEDY.csv", "ADAPTIVEm95_ER_GREEDY.csv",  "ADAPTIVEm95_SBM_GREEDY.csv", "ADAPTIVEm95_WS_GREEDY.csv", "ADAPTIVEm95_CAROAD_GREEDY.csv", "ADAPTIVEm95_INFMAXCalTech_GREEDY.csv", "ADAPTIVEm95_YOUTUBE50_GREEDY.csv"]
PrimalDual_Trials = ["ADAPTIVEm95_BA_Primal-Dual.csv", "ADAPTIVEm95_ER_Primal-Dual.csv", "ADAPTIVEm95_SBM_Primal-Dual.csv", "ADAPTIVEm95_WS_Primal-Dual.csv", "ADAPTIVEm95_CAROAD_Primal-Dual.csv", "ADAPTIVEm95_INFMAXCalTech_Primal-Dual.csv", "ADAPTIVEm95_YOUTUBE50_Primal-Dual.csv"]
Upper_Trials = ["ADAPTIVEm95_BA_UPPER_BOUNDS.csv", "ADAPTIVEm95_ER_UPPER_BOUNDS.csv", "ADAPTIVEm95_SBM_UPPER_BOUNDS.csv", "ADAPTIVEm95_WS_UPPER_BOUNDS.csv", "ADAPTIVEm95_CAROAD_UPPER_BOUNDS.csv", "ADAPTIVEm95_INFMAXCalTech_UPPER_BOUNDS.csv", "ADAPTIVEm95_YOUTUBE50_UPPER_BOUNDS.csv"]

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
    dualfit_vals = []
    DUAL_vals = []
    topk_vals = []
    marginal_vals = []
    curvature_vals = []
    pd_vals = []
    k_vals = []
    n = 0
    with open("experiment_results_output_data/"+Greedy_Trials[i]) as greedycsv:
      reader_greedy = csv.reader(greedycsv, delimiter =',')
      next(reader_greedy)
      for row in reader_greedy:
        greedy_vals.append(float(row[0]))
        k_vals.append(int(row[3]))
        dualfit_vals.append(float(row[6]))
        n = int(row[4])
    with open("experiment_results_output_data/"+PrimalDual_Trials[i]) as pdcsv:
       reader_pd = csv.reader(pdcsv, delimiter =',')
       next(reader_pd)
       for row in reader_pd:
           pd_vals.append(float(row[0]))
           dual_vals.append(float(row[3]))
    with open("experiment_results_output_data/"+Upper_Trials[i]) as pdcsv:
       reader_pd = csv.reader(pdcsv, delimiter =',')
       next(reader_pd)
       for row in reader_pd:
           DUAL_vals.append(float(row[0]))
           topk_vals.append(float(row[1]))
           marginal_vals.append(float(row[2]))
           curvature_vals.append(float(row[3]))
    X = k_vals
    ax1 = int(i / 4)
    ax2 = i % 4

    # ax[ax1, ax2].plot(X, greedy_vals, color = 'r', label = 'Greedy', linestyle='-', marker='o')
    # ax[ax1, ax2].plot(X, pd_vals, color = 'purple', label = 'Primal-Dual', linestyle='-', marker='x')
    # ax[ax1, ax2].plot(X, [l / j for l, j in zip(greedy_vals, dual_vals)], color = 'r', label = 'greedy/'+'$\sf{dual}$', linestyle='-', marker='o')
    # ax[ax1, ax2].plot(X, [l / j for l, j in zip(pd_vals, dual_vals)], color = 'purple', label = 'primal-dual/'+'$\sf{dual}$', linestyle='-', marker='x')
    # ax[ax1, ax2].plot(X, [l / j for l, j in zip(greedy_vals, dualfit_vals)], color = 'r', label = 'greedy/'+'$\sf{dualfit}$', linestyle='--', marker='^')
    # ax[ax1, ax2].plot(X, [l / j for l, j in zip(pd_vals, dualfit_vals)], color = 'purple', label = 'primal-dual/'+'$\sf{dualfit}$', linestyle='--', marker='s')
    # ax[ax1, ax2].set(xlabel='k Value', ylabel='Objective Value')
    # ax[ax1, ax2].set_title(Trial_Names[i] + " (n=" +str(n)+")")

    ax1 = int(i / 4)
    ax2 = i % 4
    # fig = plt.figure(figsize=(12,6))
    # ax = plt.subplot(111)
    X = k_vals
    ax[ax1, ax2].plot(X, [l / j for l, j in zip(greedy_vals, dual_vals)], color = 'purple', label = '$\sf{dual}$', linestyle='-', marker='o')
    ax[ax1, ax2].plot(X, [l / j for l, j in zip(greedy_vals, dualfit_vals)], color = 'orange', label = '$\sf{dualfit}$', linestyle='-', marker='x')
    ax[ax1, ax2].plot(X, [l / j for l, j in zip(greedy_vals, DUAL_vals)], color = 'g', label = 'BQSDUAL', linestyle='--', marker='^')
    ax[ax1, ax2].plot(X, [l / j for l, j in zip(greedy_vals, topk_vals)], color = 'r', label = 'topk', linestyle='--', marker='s')
    ax[ax1, ax2].plot(X, [l / j for l, j in zip(greedy_vals, marginal_vals)], color = 'b', label = 'marginal', linestyle='--', marker='p')
    ax[ax1, ax2].plot(X, curvature_vals, color = 'brown', label = 'curvature', linestyle='--', marker='*')
    ax[ax1, ax2].set(xlabel='k Value', ylabel='Approximation Bound')
    ax[ax1, ax2].set_title(Trial_Names[i] + " (n=" +str(n)+")")

    # dualfit_avgs.append(mean([l / j for l, j in zip(greedy_vals, dualfit_vals)]))
    # topk_avgs.append(mean([l / j for l, j in zip(greedy_vals, topk_vals)]))
    # marginal_avgs.append(mean([l / j for l, j in zip(greedy_vals, marginal_vals)]))
    # curvature_avgs.append(mean(curvature_vals))
    print(Trial_Names[i])
    print("Greedy/Dual", mean([l / j for l, j in zip(greedy_vals, dual_vals)]))
    print("Primal-Dual/Dual", mean([l / j for l, j in zip(pd_vals, dual_vals)]))
    print("Greedy/DualFit", mean([l / j for l, j in zip(greedy_vals, dualfit_vals)]))
    print("Primal-Dual/DualFit", mean([l / j for l, j in zip(pd_vals, dualfit_vals)]))
    print("Greedy/DUAL", mean([l / j for l, j in zip(greedy_vals, DUAL_vals)]))
    print("-------------------------------------------------------")
    print("-------------------------------------------------------")
figure.delaxes(ax[1, 3])
ax[1][2].legend(loc='center left', bbox_to_anchor=(1.3, 0.5))
plt.show()
print(dualfit_avgs)
print(topk_avgs)
print(marginal_avgs)
print(curvature_avgs)