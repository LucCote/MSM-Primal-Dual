import csv
import matplotlib.pyplot as plt

Trials = ["ADAPTIVEm95_BA_TIMETRIAL2.csv", "ADAPTIVEm95_ER_TIMETRIAL2.csv",  "ADAPTIVEm95_SBM_TIMETRIAL2.csv", "ADAPTIVEm95_WS_TIMETRIAL2.csv", "ADAPTIVEm95_CAROAD_TIMETRIAL2.csv", "ADAPTIVEm95_INFMAXCalTech_TIMETRIAL2.csv", "ADAPTIVEm95_YOUTUBE50_TIMETRIAL2.csv"]

Trial_Names = ["BA Graph", "ER Graph", "SBM Graph", "WS Graph", "CA ROADs Dataset", "Caltech Facebook Dataset", "Youtube Reccomendation Dataset"]

figure, ax = plt.subplots(2, 4)

dual_avgs = []
DUAL_avgs = []
GREEDY_avgs = []

def mean(data):
  return sum(data)/len(data)

for i in range(len(Trials)):
    valG_vec = []
    valPD_vec = []
    dual_vec = []
    valD_vec = []
    queriesG_vec = []
    queriesPD_vec = []
    queriesD_vec = []
    timeG_vec = []
    timePD_vec = []
    timeD_vec = []
    k_vec = []
    n = 0
    # dual,DUAL,greedy,pd,queriesG,queriesPD,queriesD,timeG,timePD,timeD,k,n,trial
    with open("experiment_results_output_data/"+Trials[i]) as greedycsv:
      reader_greedy = csv.reader(greedycsv, delimiter =',')
      next(reader_greedy)
      for row in reader_greedy:
        valG_vec.append(float(row[2]))
        valPD_vec.append(float(row[3]))
        valD_vec.append(float(row[1]))
        dual_vec.append(float(row[0]))
        queriesG_vec.append(float(row[4]))
        queriesPD_vec.append(float(row[5]))
        queriesD_vec.append(float(row[6]))
        timeG_vec.append(float(row[7]))
        timePD_vec.append(float(row[8]))
        timeD_vec.append(float(row[9]))
        k_vec.append(float(row[10]))
        n = int(row[11])
    # X = [k for k in range(len(greedy_vals))]
    ax1 = int(i / 4)
    ax2 = i % 4
    X = k_vec
    ax[ax1, ax2].plot(X, timePD_vec, color = 'purple', label = 'primal-dual', linestyle='-', marker='o')
    ax[ax1, ax2].plot(X, timeD_vec, color = 'g', label = 'BQSDUAL', linestyle='-', marker='^')
    ax[ax1, ax2].plot(X, timeG_vec, color = 'r', label = 'greedy', linestyle='-', marker='s')
    ax[ax1, ax2].set(xlabel='k Value', ylabel='Time (seconds)')
    ax[ax1, ax2].set_title(Trial_Names[i] + " (n=" +str(n)+")")

    # fig = plt.figure(figsize=(12,6))
    # ax = plt.subplot(111)
    # X = k_vec
    # ax.plot(X, queriesPD_vec, color = 'g', label = 'Primal Dual', linestyle='-')
    # ax.plot(X, queriesD_vec, color = 'b', label = 'DUAL', linestyle='-')
    # ax.plot(X, queriesG_vec, color = 'r', label = 'Greedy', linestyle='-')
    # plt.xlabel("k Value")
    # plt.ylabel("Queries")
    # plt.title(Trial_Names[i] + " (n=" +str(n)+")")
    # # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()  

    # fig = plt.figure(figsize=(12,6))
    # ax = plt.subplot(111)
    # X = k_vec
    # ax.plot(X, valG_vec, color = 'g', label = 'Primal Dual', linestyle='-')
    # ax.plot(X, valD_vec, color = 'b', label = 'DUAL', linestyle='-')
    # plt.xlabel("k Value")
    # plt.ylabel("Value")
    # plt.title(Trial_Names[i] + " (n=" +str(n)+")")
    # # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()  

    # print(Trial_Names[i])
    # print("Greedy/Dual", mean([l / j for l, j in zip(greedy_vals, dual_vals)]))
    # print("Primal-Dual/Dual", mean([l / j for l, j in zip(pd_vals, dual_vals)]))
    # print("Greedy/DualFit", mean([l / j for l, j in zip(greedy_vals, dualfit_vals)]))
    # print("Primal-Dual/DualFit", mean([l / j for l, j in zip(pd_vals, dualfit_vals)]))
    # print("-------------------------------------------------------")

figure.delaxes(ax[1, 3])
ax[1][2].legend(loc='center left', bbox_to_anchor=(1.3, 0.5))
plt.show()