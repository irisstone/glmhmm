### Description of the Data

The data contained in this folder is behavioral data from mice performing a two alternative forced choice (2AFC) task in which the animals run down a virtual maze while multi-sensory "cues" appear to their left and right. The mice must "accumulate evidence" as these cues appear and ultimately make a decision to turn left or right based on which side of the maze had more cues.

The datasets in the paper include three cohorts of mice: a group that was inhibited in the direct pathway of the striatum, a group that was inhibited in the indirect pathway, and a control (no opsin) group. This folder contains the following information for each cohort:

#### x
Design matrices consisting of all the external covariates used in model regressions, including (in order from the first to last column):
* <b>bias:</b> the offset or intercept term
* <b>delta cues:</b> the standardized difference in the number of cues that appear on the right (R) and left (L) sides of the maze (coded as R-L)
* <b>laser:</b> the on/off status of the laser on each trial, indicating whether striatal inhibition occured
* <b>previous choices 1-6:</b> the choice that the animal made on the previous trial (and up to six trials ago)
* <b>previous rewarded choice</b> the choice that the animal made on the previous trial and whether or not that choice was rewarded
     
See the methods section of the paper for more information on how we coded these covariates. 

#### y
The choice behavior for each mouse on each trial (-1 for left choices and 1 for right choices)

#### dates
Date information indicating whether each trial was conducted on an odd numbered date (denoted by a -1), or an even numbered date (denoted by a 1).

#### sessions 
The starting index of each new session (corresponding to different days; roughly 190 trials on average per session).

#### mouseIDs
An ID on each trial indicating which mouse was performing the task on that trial.
