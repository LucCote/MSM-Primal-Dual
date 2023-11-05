### Primal-Dual Monotone Submodular Maximization
This repository contains the code for the experiments in "Primal-Dual Algorithm for Monotone Submodular Maximization under Cardinality Constraints" by Deeparnab Chakrabarty and Luc Cote. All experiments were run on a linode compute server using an anaconda environment. Experimental results can be found in the [experiment_results_output_data](./experiment_results_output_data) folder, and graphs summarizing these results can be found in the [visuals](./visuals) folder.

### Acknowledgement
This code was adapted from the submodular library written by Adam Breuer. The readme from that repository is included below. 

### submodular ###
*submodular* is a python library of **high-performance MPI-parallelized implementations of state-of-the-art algorithms for submodular maximization** written by **Adam Breuer**. It also includes an **experimental comparison framework** to compare algorithms, as well as **additional simple-to-run serial (non-parallel) implementations**. 

When citing this libary, please cite our paper:
https://arxiv.org/abs/1907.06173

 - Breuer, A., Balkanski, E., & Singer, Y. "The FAST Algorithm for Submodular Maximization". *International Conference on Machine Learning (ICML) 2020.*

### Getting Started with MPI parallel computing, Amazon AWS, and the submodular library: ###
I'm happy to provide a quick **tutorial and examples for running *submodular* on AWS in parallel** or to discuss how to **use *submodular* for your research**. Email me at **breuer `at' harvard.edu**

### Replicating our Experiments: ###
**!NOTE! that you will need to download the data file *Movie_ratings_mat.csv* and place it in the *submodular/data* directory**. See www.adambreuer.com/code.
Our experiments can be replicated by running the following scripts in parallel:
 To replicate *Experiments set 1:*
 - *run_experiments_set1_adaptive.py*    

To replicate *Experiments set 2:*
- *run_experiments_set2_ltlg.py*   

Results data will be automatically saved as CSV files in the *experiment_results_output_data* directory.


### Classic Submodular Maximization Algorithms: ###

**Top K** (Returns the solution containing the top k elements with the highest marginal contribution to the empty set.).
  - topk()
  - topk_parallel()

**Random** (Returns the value of a random set of size k.).
  - randomksolutionval()
  - randomksolutionval_parallel()

**Greedy** (Nemhauser 1978).
  - greedy()
  - greedy_parallel()
    
**Lazy Greedy** (Minoux 1978).
  - lazygreedy()



### State-of-the-art Submodular Maximization Algorithms: ###

**FAST** (Breuer, Balkanski, and Singer, ICML 2020).
  - FAST_knowopt() -- runs FAST (non-parallel) for a single guess of the optimal solution value (OPT)
  - FAST_guessopt() -- runs FAST (non-parallel) when the optimal solution value (OPT) is unknown
  - FAST_knowopt_parallel() -- runs FAST in parallel for a single guess of the optimal solution value (OPT)
  - FAST_guessopt_parallel() -- runs FAST in parallel when the optimal solution value (OPT) is unknown

**Stochastic Greedy** (Mirzasoleiman et al., 2015).
  - stochasticgreedy()
  - stochasticgreedy_parallel()

**Lazier Than Lazy Greedy** (Mirzasoleiman et al., 2015).
  - lazierthanlazygreedy()
  - lazierthanlazygreedy_parallel()

**Amortized Filtering** (Balkanski et al., SODA 2019).
  - amortizedfiltering()
  - amortizedfiltering_parallel()
  - amortizedfilteringOPTS() -- runs amortizedfiltering over various guesses for OPT
  - amortizedfilteringOPTS_parallel() -- runs amortizedfiltering_parallel over various guesses for OPT

**Adaptive Sequencing** (Balkanski, et al., STOC 2019).
  - adapt_sequencing_knowopt_parallel()
  - adapt_sequencing_guessopt_parallel() -- runs Adaptive Sequencing in parallel when the optimal solution value (OPT) is unknown

**Exhaustive Maximization** (Fahrbach et al. Algorithm 3 from "Submodular Maximization with Nearly Optimal Approximation, Adaptivity and Query Complexity").
  - exhaustivemax()
  - exhaustivemax_parallel()

**Binary Search Maximization** (Fahrbach et al. Algorithm 4 from "Submodular Maximization with Nearly Optimal Approximation, Adaptivity and Query Complexity").
  - binseachmax_parallel()

**Randomized Parallel Greedy** (Chekuri et al. 2019, randomized-parallel-greedy for cardinality constraints algorithm from Figure 2 p. 12 of "Submodular Function Maximization in Parallel via the Multilinear Relaxation").
  - randparagreedy()
  - randparagreedy_parallel()
  - randparagreedyOPTS() -- runs randparagreedy() over multiple guesses for OPT
  - randparagreedyOPTS_parallel() -- runs randparagreedy_parallel() over multiple guesses for OPT



### Objective functions: ###
We include several canonical submodular objective functions, such as:
- **Movie Recommendation;**
- **Network Max Cover;**
- **Revenue Maximization on YouTube;**
- **Influence Maximization;** 
- **Traffic Sensor Placement.**




