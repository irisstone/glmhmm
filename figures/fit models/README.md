This folder contains saved model objects generated when fitting GLMs and GLM-HMMs to real and simulated data in service of recreating all the plots described in [Bolkan, Stone et al 2021](https://www.biorxiv.org/content/10.1101/2021.07.23.453573v1). See the <code>figures</code> folder for the fitting code that produces the model objects (and associated inferred parameters) saved in this folder. 

The saved models include: 

* GLM_d1.pickle: GLM fit to data from the direct pathway cohort (see <code>fig4.ipynb</code>, <code>fig5.ipynb</code>)
* GLM_d2.pickle: GLM fit to data from the indirect pathway cohort (see <code>fig4.ipynb</code>, <code>fig5.ipynb</code>)
* GLMHMM_d1.pickle: 3-state GLM-HMM fit to data from the direct pathway cohort (see <code>fig5.ipynb</code>, <code>fig6.ipynb</code>, <code>fig7.ipynb</code>, <code>extdatafig9.ipynb</code>, <code>suppfig4.ipynb</code>)
* GLMHMM_d2.pickle: 3-state GLM-HMM fit to data from the indirect pathway cohort (see <code>fig5.ipynb</code>, <code>fig6.ipynb</code>, <code>fig7.ipynb</code>, <code>extdatafig9.ipynb</code>, <code>suppfig4.ipynb</code>)
* simulated_GLMHMMs_d1.pickle: 3-state GLM-HMM fit to simulated data, which was generated using the fit model params for the model fit to the real data from the direct pathway cohort (see <code>extdatafig9.ipynb</code>)
* simulated_GLMHMMs_d2.pickle: 3-state GLM-HMM fit to simulated data, which was generated using the fit model params for the model fit to the real data from the indirect pathway cohort (see <code>extdatafig9.ipynb</code>)
* training_states_GLMHMMs_d1.pickle: K-state GLM-HMMs fit to each of five training datasets from the direct pathway cohort for K={1,2,3,4,5}, with a 1-state GLM-HMM equivalent to a GLM (see <code>extdatafig7.ipynb</code>)
* training_states_GLMHMMs_d2.pickle: K-state GLM-HMMs fit to each of five training datasets from the indirect pathway cohort for K={1,2,3,4,5}, with a 1-state GLM-HMM equivalent to a GLM (see <code>extdatafig7.ipynb</code>)
* training_states_GLMHMMs_ct.pickle: K-state GLM-HMMs fit to each of five training datasets from the control cohort for K={1,2,3,4,5}, with a 1-state GLM-HMM equivalent to a GLM (see <code>extdatafig7.ipynb</code>)