This folder contains several jupyter notebooks that walk users through the details of how to fit GLMs and GLM-HMMs to the real data described in [Bolkan, Stone et al 2021](https://www.biorxiv.org/content/10.1101/2021.07.23.453573v1). These notebooks recreate all the plots associated with the modeling results in the paper and provide a useful starting point for users who might want to run similar analyses on their own data. 

#### folders
The <code>data</code> folder contains all the relevant data to reproduce the plots in a manner slightly processed from its original form (e.g. reformatted from .mat files into .npy files that are readable as matrices in python). The data in its original form (as well as some additional data collected during the task) is available on figshare [here](https://figshare.com/search?q=10.6084%2Fm9.figshare.17299142). To see other plots from the paper, check out my co-author Scott Bolkan's [github repository](https://github.com/ssbolkan/BolkanStoneEtAl). 

The <code>fit models</code> folder contains the model objects (with fitted parameters stored in the objects) from several models fit during the figure generation process. To avoid having to refit the same models several times when running different notebooks, we save the fit model objects here the first time we fit each one and load them in subsequent notebooks in order to quickly extract the relevant inferred parameters and save time. 

The <code>matlab</code> folder contains a small number of matlab scripts that were used to plot the psychometric curves shown in the paper. There's no need to run any scripts in this folder yourself (unless you want to). Everythiing contained in that folder is run automatically through the jupyter notebooks, using the MATLAB Engine API for Python. 

#### figures
Each jupyter notebook recreates the plots from a specified figure in [Bolkan, Stone et al 2021](https://www.biorxiv.org/content/10.1101/2021.07.23.453573v1)
* fig4.ipynb: a notebook describing how to fit a GLM to real data and recreating the plots from Figure 4
* fig5.ipynb: a notebook comparing model performance between a standard Bernoulli GLM and a 3-state GLM-HMM, as seen in Figure 5
* fig6.ipynb: a notebook describing how to fit a 3-state GLM-HMM to real data and recreating the plots from Figure 6
* fig7.ipynb: a notebook demonstrating statistical analyses to describe how the three states identified by the GLM-HMM manifest in the data and recreating the plots from Figure 7
* extdatafig7: a notebook detailing model selection and control analyses and recreating the plots from Extended Data Figure 7
* extdatafig8: a notebook demonstrating how model simulations recapitulate characteristics of the real data and recreating the plots from Extended Data Figure 8
* extdatafig9: a notebook showing how individual mice occupy different states for each session that they participate in the task and recreating the plots from Extended Data Figure 9