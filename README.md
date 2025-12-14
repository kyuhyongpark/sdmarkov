# sdmarkov - Succession diagram Markov chains

This repository is a supplement to the paper **Succession-diagram-based Markov chains reveal the attractor landscape of asynchronous Boolean networks** by Kyu Hyong Park and Réka Albert.
It contains the code used to construct and evaluate the succession diagram Markov chains (SD Markov chains) derived from stochastic asynchronous Boolean networks. 

The SD Markov chain is a coarse-grained representation of the Boolean network. The states of the Markov chain are groups of states of the Boolean network. Specifically, each state of the SD Markov chain is a trap space in the succession diagram of the Boolean network. The succession diagram is like a decision tree. You can read more about it in **Mapping the attractor landscape of Boolean networks with biobalm**, by Van-Giang Trinh, Kyu Hyong Park, Samuel Pastva, Jordan C Rozum, Bioinformatics, Volume 41,  btaf280 (2025).

Here, we consider that the state transition graph (STG) of the Boolean network is known. We use this STG to construct the transition probabilities in the Markov chain.  We evaluate the SD Markov chain’s ability to recapitulate the Boolean network’s attractors, attractor basins, decision transitions, and trajectories.  The evaluation metrics include precision, recall, negative predictive value (NPV), and specificity, as well as the quantitative metrics of root mean square difference (RMSD) and Kullback-Leibler Divergence (KLD). 

## Requirements 
biobalm  https://github.com/jcrozum/biobalm  
pystablemotifs https://github.com/jcrozum/pystablemotifs  
pyboolnet https://github.com/hklarner/pyboolnet  
Networkx (v2.4+) https://github.com/networkx/networkx/  

## Documentation
The `models/random_nk3`  directory lists the 100 Boolean networks used to evaluate the effectiveness of the SD Markov chain. Each file contains the Boolean functions of the nodes in bnet format. 

The  `notebooks` directory contains Jupyter Notebooks for generating random Boolean networks, for construction and basic analysis of the SD Markov chain, and for each type of evaluation of the SD Markov chain. 

The `results` directory contains the results of the analyses as .csv files.

