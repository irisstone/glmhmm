# glmhmm
This package provides flexible code for fitting Generalized Linear Models (GLMs), Hidden Markov Models (HMMs), and combination GLM-HMMs (also known as Input-Output HMMs or IO-HMMs). The GLM code include options for fitting observational data of either Bernoulli or Multinomial distributions. HMM inference is done using the Expectation Maximization (EM) algorithm, with an optional mode for performing deterministic-annealing EM (DAEM), which is useful for hard to fit datasets.   

### Package Contents

#### glmhmm
<li>
  <ol>glm.py: GLM class fitting code</ol>
  <ol>hmm.py: HMM class fitting code</ol>
  <ol>glm-hmm.py: GLM-HMM class fitting code</ol>
  <ol>init_params.py: a script for defining initialization options for different model parameters</ol>
  <ol>observations.py: a script for defining distribution options for GLM observations</ol>
  <ol>utils.py: a script containing miscellaneous helper functions (still being updated)</ol>
  <ol>visualize.py: a script containing plotting and post-fitting analysis functions (still being updated)</ol>
</li>

#### examples
<li>
  <ol>fit-glm.ipynb: a simple example of fitting a GLM to simulated data</ol>
  <ol>fit-hmm.ipynb: a simple example of fitting an HMM to simulated data</ol>
  <ol>fit-hmm-DAEM.ipynb: an example illustrating the benefits of deterministic annealing EM (DAEM), with a comparison to traditional EM</ol>
  <ol>fit-glm-hmm.ipynb: two examples of fitting GLM-HMMs, one to simulated data and one to real data, and in the latter case recreating some of the plots shown in [Bolkan,    Stone et al 2021](https://www.biorxiv.org/content/10.1101/2021.07.23.453573v1)) </ol>
</li>

#### data
The data folder includes pre-formatted design matrices and choice behavior for fitting the three mouse data sets described in [Bolkan, Stone et al 2021](https://www.biorxiv.org/content/10.1101/2021.07.23.453573v1)). 

### Installation
For easy installation, download the code using the link above or type the following into a terminal window:
```
git clone https://github.com/irisstone/glmhmm.git
```
For convenience, we recommend setting up a virtual environment before running the code, to avoid any unpleasant version control issues or interactions with other projects you're working on. See the env.yml file for configuration details. Note the package requires python 3.7 to run.  
