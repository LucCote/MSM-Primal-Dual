"""
@author: Luc Cote
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

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
    valGLP_vec = []
    queriesG_vec = []
    queriesPD_vec = []
    queriesD_vec = []
    queriesGLP_vec = []
    timeG_vec = []
    timePD_vec = []
    timeD_vec = []
    timeGLP_vec = []
    k_vec = []
    n = 0
    # dual,DUAL,greedy,pd,queriesG,queriesPD,queriesD,timeG,timePD,timeD,k,n,trial
    with open("../experiment_results_output_data/"+Trials[i]) as greedycsv:
      reader_greedy = csv.reader(greedycsv, delimiter =',')
      next(reader_greedy)
      valG_vec_sub = []
      valPD_vec_sub = []
      dual_vec_sub = []
      valD_vec_sub = []
      valGLP_vec_sub = []
      queriesG_vec_sub = []
      queriesPD_vec_sub = []
      queriesD_vec_sub = []
      queriesGLP_vec_sub = []
      timeG_vec_sub = []
      timePD_vec_sub = []
      timeD_vec_sub = []
      timeGLP_vec_sub = []
      k_vec_sub = []
      for row in reader_greedy:
        valG_vec_sub.append(float(row[2]))
        valPD_vec_sub.append(float(row[3]))
        valD_vec_sub.append(float(row[1]))
        dual_vec_sub.append(float(row[0]))
        valGLP_vec_sub.append(float(row[4]))
        queriesG_vec_sub.append(float(row[5]))
        queriesPD_vec_sub.append(float(row[6]))
        queriesD_vec_sub.append(float(row[7]))
        queriesGLP_vec_sub.append(float(row[8]))
        timeG_vec_sub.append(float(row[9]))
        timePD_vec_sub.append(float(row[10]))
        timeD_vec_sub.append(float(row[11]))
        timeGLP_vec_sub.append(float(row[12]))
        k_vec_sub.append(float(row[13]))
        if len(k_vec_sub) % 5 == 0:
          valG_vec.append(np.median(valG_vec_sub))
          valPD_vec.append(np.median(valPD_vec_sub))
          valD_vec.append(np.median(valD_vec_sub))
          dual_vec.append(np.median(dual_vec_sub))
          valGLP_vec.append(np.median(valG_vec_sub))
          queriesG_vec.append(np.median(queriesG_vec_sub))
          queriesPD_vec.append(np.median(queriesPD_vec_sub))
          queriesD_vec.append(np.median(queriesD_vec_sub))
          queriesGLP_vec.append(np.median(queriesG_vec_sub))
          timeG_vec.append(np.median(timeG_vec_sub))
          timePD_vec.append(np.median(timePD_vec_sub))
          timeD_vec.append(np.median(timeD_vec_sub))
          timeGLP_vec.append(np.median(timeGLP_vec_sub))
          k_vec.append(np.median(k_vec_sub))
        n = int(row[14])
    
    ### Time vs. K value
    ax1 = int(i / 4)
    ax2 = i % 4
    X = k_vec

    ax[ax1,ax2].plot(X, timePD_vec, color = 'purple', label = 'Primal-Dual', linestyle='-', marker='o')
    ax[ax1,ax2].plot(X, timeD_vec, color = 'g', label = 'BQS', linestyle='-', marker='^')
    ax[ax1,ax2].plot(X, timeG_vec, color = 'r', label = 'Greedy', linestyle='-', marker='s')
    ax[ax1,ax2].plot(X, timeGLP_vec, color = 'b', label = 'GreeDual2', linestyle='-', marker='x')
    ax[ax1,ax2].set(xlabel='k Value', ylabel='Time (seconds)')
    ax[ax1,ax2].set_title(Trial_Names[i] + " (n=" +str(n)+")")

figure.delaxes(ax[1, 3])
ax[1][2].legend(loc='center left', bbox_to_anchor=(1.3, 0.5))
plt.show()