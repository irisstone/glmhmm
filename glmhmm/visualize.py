#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 17:20:42 2021

@author: irisstone

Functions for visualizing and plotting results related to glmhmm fitting code
"""
import matplotlib.pyplot as plt
import numpy as np
import warnings
from glmhmm.utils import find_best_fit, uniqueSessionIDs
from glmhmm.analysis import fit_line_to_hist, dwell_times_per_session, session_lengths_for_animal
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# set plot design features
font = {'family'   : 'sans-serif',
        'sans-serif' : 'Helvetica',
        'weight'   : 'regular',
        'size'     : 24}

mpl.rc('font', **font)
    
def plot_model_params(M,ax,precision='%.2f'):
    
    # plot heat map of transitions
    ax.imshow(M,cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # add numerical values to plot
    I = M.shape[0]
    J = M.shape[1]
    for i in range(I):
        for j in range(J):
            if M[i,j] >= 0.5:
                color='black'
            else:
                color='white'

            if J > 1:
                ax.text((j+1)/(J)-(1/(J+1)),((I-i)/I)-(1/(I+2)),precision %(M[i,j]),transform=ax.transAxes,fontsize=15,color=color)
            else:
                ax.text(0.3,((I-i)/I)-(1/(I+2)),precision %(M[i,j]),transform=ax.transAxes,fontsize=15,color=color)

    ax.set_xlabel('state t+1')
    ax.set_ylabel('state t')
    ax.set_xticks(np.arange(M.shape[0]))
    ax.set_xticklabels(np.arange(1,M.shape[0]+1))
    ax.set_yticks(np.arange(M.shape[0]))
    ax.set_yticklabels(np.arange(1,M.shape[0]+1))

def plot_loglikelihoods(lls,maxdiff,ax,startix=5):
    '''
    Plot the trajectory of the log-likelihoods for multiple fits, identify how many top fits (nearly) match, and 
    color those trajectories in the plot accordingly
    '''
    
    # get the final ll for each fit
    final_lls = np.array([np.amax(lls[i,~np.isnan(lls[i,:])]) for i in range(lls.shape[0])])
    
    # get the index of the top ll
    bestix = find_best_fit(lls)
    
    # compute the difference between the top ll and all final lls
    ll_diffs = final_lls[bestix] - final_lls
    
    # identify te fits where the difference from the top ll is less than maxdiff
    top_matching_lls = lls[ll_diffs < maxdiff,:]
    
    # plot
    ax.plot(np.arange(startix,lls.shape[1]),lls.T[startix:], color='black')
    ax.plot(top_matching_lls.T[startix:], color='red')
    ax.set_xlabel('iterations of EM', fontsize=16)
    ax.set_ylabel('log-likelihood', fontsize=16)
    
    return np.where(ll_diffs < maxdiff)[0] # return indices of best (matching) fits


def plot_weights(w,ax,xlabels=None,color=None,style='-',label=[''],switch=False, error=None):
    
    if switch:
        w = np.insert(w,3,w[:,0],axis=1)
        w = np.delete(w,0,axis=1)
    
    if color is not None:
        if error is not None:
            error = error[(w.shape[0])*(w.shape[0]-1):]
            error = np.reshape(error,(w.shape[0],w.shape[1]))
            for i in range(w.shape[0]):
                ax.errorbar(np.arange(w[i,:].shape[0]),w[i,:],yerr=error[i,:],fmt=style,color=color[i],label=label[i],linewidth=2)
        else:
            for i in range(w.shape[0]):
                ax.plot(w[i,:],style,color=color[i],label=label[i],linewidth=2)
    else:
        if error is not None:
            ax.errorbar(w.T,yerr=error,fmt=style,label=label)
        else:
            ax.plot(w.T,style,label=label)
    ax.set_ylabel('weight')
    if xlabels:
        ax.plot(xlabels,np.zeros((len(xlabels),1)),'k--')
        ax.set_xticks(np.arange(0,len(xlabels)))
        ax.set_xticklabels(xlabels,rotation=90)
        
def plot_psychometrics(colors,title,file_path,save_path,lb=None,ub=None):

    if lb is None: 
        lb_off = [1e-6, 1e-6, -np.inf, -np.inf]
        lb_on = [1e-6, 1e-6, -np.inf, -np.inf]
    if ub is None: 
        ub_off = [0.75, 0.75, np.inf, np.inf]
        ub_on = [0.75, 0.75, np.inf, np.inf]
    else:
        lb_off = lb[0]
        lb_on = lb[1]
        ub_off = ub[0]
        ub_on = ub[1]

    import matlab.engine
    eng = matlab.engine.start_matlab()
    s = eng.genpath('matlab')
    eng.addpath(s, nargout=0)

    # convert to matlab data types
    lb = matlab.double([lb_off,lb_on])
    ub = matlab.double([ub_off,ub_on])
    colors = matlab.double(colors)
    title = eng.convertCharsToStrings(title)
    file_path = eng.convertCharsToStrings(file_path)
    save_path = eng.convertCharsToStrings(save_path)

    ret = eng.fit_psychometrics(colors,title,file_path,save_path,lb,ub)

def plot_glmvsglmhmm_performance(data,label,color,avg_sess_length,ax,axis_len=80):
    ax.plot(np.arange(0,axis_len,0.001),np.arange(0,axis_len,0.001),'k--', linewidth=3)
    ax.plot(data[:,0]*avg_sess_length,data[:,1]*avg_sess_length,'o',markersize=10,color=color,label=label)

    ax.set_xlim([0.0,axis_len])
    ax.set_ylim([0.0,axis_len])
    ax.set_xticks(np.arange(10,75,20))
    ax.set_yticks(np.arange(10,75,20))
    ax.set_xticklabels(np.arange(10,75,20),fontsize=24)
    ax.set_yticklabels(np.arange(10,75,20),fontsize=24)
    ax.legend(fontsize=20, loc=4)

def plot_histogram_run_lengths(bin_heights,bin_edges,ax,color=[0,0,0],label=''):
    '''
    Recreates Fig 5E/F from the paper.

    Parameters
    ----------
    bin_heights : num_sims x num_bins array containing the value of each bin height of each histogram
    bin_edges : num_bins + 1 vector containing the values of the bin edges
    ax : the figure axis handle
    color : desired color for plotting, optional
    label : the label to be used in the legend, optional
    '''

    # determines whether to take average of multiple histograms based on shape of bin_heights
    if len(bin_heights.shape) > 1: 
        take_average = True
        num_bins = bin_heights.shape[1]
    else: 
        take_average = False
        num_bins = bin_heights.shape[0]

    if take_average:
        # compute statistics
        num_sims = bin_heights.shape[0]
        avg_bin_heights = np.mean(bin_heights,axis=0)
        std_bin_heights = np.std(bin_heights,axis=0)
        confidence_interval = avg_bin_heights + 1.96*(std_bin_heights/np.sqrt(num_sims))
        confidence_range = confidence_interval - avg_bin_heights
    else:
        avg_bin_heights = bin_heights

    # obtain smoothed curve from averaged bin heights
    smoothed_counts = fit_line_to_hist(avg_bin_heights,window_size=4)

    # plot results
    half_bin_width = (bin_edges[1]-bin_edges[0])/2
    x = np.linspace(bin_edges[0]-half_bin_width, bin_edges[-1]-half_bin_width, num_bins)
    ax.plot(x,smoothed_counts,color=color,label=label,linewidth=3)
    if take_average:
        ax.fill_between(x,smoothed_counts-confidence_range,smoothed_counts+confidence_range,color=color,alpha=0.3)
    ax.legend(fontsize=20)
    ax.set_ylabel('counts')
    ax.set_xlabel('consecutive choices contra/ipsilateral \n to inhibited hemisphere')

def plot_state_performance(y,z,trialTypes,mouseIDs,colors,ax):

    K = len(np.unique(z))
    rewarded = np.array(y==trialTypes)*1
    IDs = np.unique(mouseIDs) 
    num_mice = len(IDs)       
        
    # get number of correct choices for each state and mouse
    state_correct = np.zeros((K,num_mice))
    state_incorrect = np.zeros((K,num_mice))
    num_trials = np.zeros((K,num_mice))

    for i in range(K):
        for j in range(num_mice): 
            IDix = np.where(mouseIDs==IDs[j])[0]
            z_mouse = z[IDix] 
            rewarded_mouse = rewarded[IDix] 
            num_trials[i,j] = len(np.where(z_mouse==i)[0])
            z_mouse_state = np.where(z_mouse==i)[0] 
            state_correct[i,j] = np.sum(rewarded_mouse[z_mouse_state])
                
    num_trials[num_trials==0] = np.nan
    percent_correct = state_correct/num_trials * 100
    avg_percent_correct = np.nanmean(percent_correct,axis=1)
        
    Labels = ('state 1', 'state 2', 'state 3')
    ax.bar(Labels, np.squeeze(avg_percent_correct),color=colors)
    ax.plot(Labels,percent_correct,'ko',markersize=2)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(Labels,rotation=90,fontsize=24)
    ax.set_ylabel('correct trials (%)',fontsize=24)
    ax.set_ylim([0,100])
    ax.set_yticks([0,25,50,75,100])
    ax.set_yticklabels([0,25,50,75,100],fontsize=24)

def plot_percent_laser_trials(z,laserStatus,mouseIDs,colors,ax):

    K = len(np.unique(z))
    IDs = np.unique(mouseIDs)
    num_mice = len(IDs)  
    nlaserOFF = np.zeros((len(IDs),K))
    nlaserON = np.zeros((len(IDs),K))

    for i in range(num_mice):   
        for j in range(K):
            nlaserOFF[i,j] = len(z[(mouseIDs==IDs[i]) & (z==j) & (laserStatus==0)])
            nlaserON[i,j] = len(z[(mouseIDs==IDs[i]) & (z==j) & (laserStatus!=0)])
        
    total_trials = nlaserOFF + nlaserON
    total_trials[total_trials==0] = np.nan
    percentageON = (nlaserON/total_trials)*100
    meanpercentageON = np.nanmean(percentageON,axis=0)

    Labels = ('state 1','state 2', 'state 3')
    ax.bar(Labels,meanpercentageON,color=colors,width=0.8)
    ax.plot(Labels, percentageON.T, 'ko', markersize=2)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(Labels,rotation=90,fontsize=24)
    ax.set_yticks(np.arange(0,25,8))
    ax.set_yticklabels(np.arange(0,25,8),fontsize=24)
    ax.set_ylabel('trials in state (%)', fontsize=24)

def plot_triggered_average(z,laserStatus,colors,ax,window=5):

    K = len(np.unique(z))

    # find trials where event occurs
    event_ixs = np.where(laserStatus!=0)[0]

    # identify states before/after event
    num_back = window
    num_forward = window+1
    states_around_event = np.zeros((len(event_ixs),num_back+num_forward))

    for i in range(len(event_ixs)):
        if  len(z)-num_forward >= event_ixs[i] >= num_back:
            states_around_event[i,:] = z[event_ixs[i] - num_back: event_ixs[i] + num_forward]

    state_distributions = np.zeros((K,states_around_event.shape[1]))
    for i in range(states_around_event.shape[1]):
        for j in range(K):
            state_distributions[j,i] = len(np.where(states_around_event[:,i] == j)[0])/states_around_event.shape[0]
            
    ax.plot(np.zeros((100)),np.arange(0,100),'k--')
    for j in range(K):
        ax.plot(np.arange(-5,6,1),state_distributions[j],'.-',color = colors[j],linewidth=3, markersize=10)
        
    ax.set_ylim([0.0,0.60])
    ax.set_yticks(np.arange(0.0,0.61,0.30))
    ax.set_yticklabels(np.arange(0.0,0.61,0.30),fontsize=24)
    ax.set_xticks(np.arange(-4,5,2))
    ax.set_xticklabels(np.arange(-4,5,2),fontsize=24)
    ax.set_xlabel('# of trials from laser', fontsize=24)
    ax.set_ylabel('avg. p(state)', fontsize=24)

def plot_example_sessions(zprobs,sessions,ax,colors,session_number=0, example=1):

    '''
    Recreates Fig 7C/D from the paper.

    Parameters
    ----------
    zprobs : nxk matrix of state probabilities for each trial
    sessions : vector of the start index of each session 
    ax : the figure axis handle
    session_number : the desired session number to be plotted
    example : optional, the number example session to be plotted
    '''

    K = zprobs.shape[1] # number of states to plot

    # get the subset of the zprobs matrix corresponding to the desired session
    start = sessions[session_number]
    stop = sessions[session_number+1]
    zprobs_session = zprobs[start:stop,:]
    
    # plot p(state) for each state
    for i in range(K):
        ax.plot(np.arange(zprobs_session.shape[0]),zprobs_session[:,i],color=colors[i], label='state %s' %(i+1), linewidth=3)

    # x-axis formatting
    xlabels = np.array(np.round(np.linspace(0,stop-start-1,num=6,endpoint=True),0),dtype=int)
    ax.set_xlim([0,stop-start-1])
    ax.set_xticks(xlabels)
    ax.set_xticklabels(xlabels,fontsize=24)
    ax.set_xlabel("trials within session", fontsize=24)

    # y-axis formatting
    ylabels = [0,0.5,1]
    ax.set_ylim(0,1)
    ax.set_yticks(ylabels)
    ax.set_yticklabels(ylabels, fontsize=24)
    ax.set_ylabel('p(state)', fontsize=24)

    ax.set_title('example session %s' %(example),fontsize=24)

def plot_average_state_probabilities(zprobs,sessions,colors,axes,alpha=1,linewidth=3):

    K = zprobs.shape[1] # number of states to plot
    session_lengths = np.diff(sessions)
    first50trials = np.empty((0,50,zprobs.shape[1]))
    last50trials = np.empty((0,50,zprobs.shape[1]))

    for i in range(len(session_lengths)):

        # get the subset of the state probabilities corresponding to each session
        start = sessions[i]
        stop = sessions[i+1]
        zprobs_session = zprobs[start:stop,:]

        if len(zprobs_session) >= 100: # session has to be at least 100 trials long to be included

            first50trials = np.concatenate((first50trials,zprobs_session[np.newaxis,0:50,:]),axis=0)
            last50trials = np.concatenate((last50trials,zprobs_session[np.newaxis,-50:,:]),axis=0)


    avg_first50 = np.mean(first50trials,axis=0)
    avg_last50 = np.mean(last50trials,axis=0)
    averages = [avg_first50,avg_last50]

    # plot
    for j in range(2):
        for i in range(K):
            axes[j].plot(averages[j][:,i],label='state %s' %(i+1), color=colors[i],linewidth=linewidth,alpha=alpha)
        axes[j].set_xlim([0,50])
        axes[j].set_xticks(np.arange(0,60,25))
        axes[j].set_xticklabels(np.arange(0,60,25), fontsize=24)
        axes[j].set_yticks(np.arange(0,0.8,0.3))
        if j == 0:
            axes[j].set_ylabel('avg. p(state)', fontsize=24)
            axes[j].set_xlabel('first 50 trials', fontsize=24)
            axes[j].set_yticklabels(np.arange(0,0.8,0.3),fontsize=24)
        else:
            axes[j].set_yticklabels([])
            axes[j].set_yticks([])
            axes[j].set_xlabel('last 50 trials', fontsize=24)

def plot_average_dwell_time(z,sessions,mouseIDs,colors,ax,terminal_run=False):

    K = len(np.unique(z)) # number of states to plot 
    unique_mouse_IDs = np.unique(mouseIDs) # vector of length equal to the number of mice
    session_IDs = uniqueSessionIDs(sessions) # vector of length N assigning each trial a unique session ID
    average_run_length_per_state_per_mouse = np.zeros((len(unique_mouse_IDs),K))

    # loop through each mouse
    for i in range(len(unique_mouse_IDs)):
        mouse_ixs,session_lengths_mouse = session_lengths_for_animal(mouseIDs,unique_mouse_IDs[i],session_IDs)
        z_mouse = z[mouse_ixs] # get the state assignments for that mouse

        # loop through each session for each mouse
        start = 0
        run_lengths_in_session  = [[] for i in range(K)] # initialize empty list for each state
        for j in range(len(session_lengths_mouse)):
            runStates = z_mouse[start:start+session_lengths_mouse[j]] # get states for session  
            # get run lengths for each session 
            run_lengths_in_session = dwell_times_per_session(runStates,dwell_times=run_lengths_in_session,terminal_run=terminal_run)        
            start += session_lengths_mouse[j]

        # get average run length for each state
        average_run_length_per_state = np.zeros((K))
        for m in range(len(run_lengths_in_session)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) # suppress warning when a run length is empty
                average_run_length_per_state[m] = np.nanmean(np.array(run_lengths_in_session[m]))
            
        average_run_length_per_state_per_mouse[i,:] = average_run_length_per_state
    
    average_run_length_across_mice = np.nanmean(average_run_length_per_state_per_mouse,axis=0)

    # plot
    Labels = ('state 1','state 2', 'state 3')
    ax.bar(Labels,average_run_length_across_mice,color=colors, width=0.8)
    ax.plot(Labels, average_run_length_per_state_per_mouse.T, 'ko', markersize=2)
    ax.set_ylabel('trials')
    ax.set_yticks([0,60,120,180])
    ax.set_yticklabels([0,60,120,180])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(Labels,rotation=90)

def plot_fraction_of_trials_per_state(zprobs,sessions,mouseIDs,colors,ax,mouse=None):

    K = zprobs.shape[1] # number of states to plot
    session_IDs = uniqueSessionIDs(sessions) # vector of length N assigning each trial a unique session ID
      
    # get mouse IDs so they are sorted in the same order as they appear in the dataset
    ids, idixs = np.unique(mouseIDs,return_index=True)
    sorted_mouse_IDs = ids[np.argsort(idixs)]

    if mouse is not None:
        sorted_mouse_IDs = [sorted_mouse_IDs[mouse]]

    # initialize empty arrays and lists
    prop_time_in_each_state_all_mice = np.empty((K))
    prop_time_in_each_state_average_mice = np.empty((len(sorted_mouse_IDs),K))

    for i in range(len(sorted_mouse_IDs)): # for each mouse
        mouse_ixs,session_lengths_mouse = session_lengths_for_animal(mouseIDs,sorted_mouse_IDs[i],session_IDs)
        zprobs_mouse = zprobs[mouse_ixs]
    
        start = 0
        prop_time_in_each_state = np.zeros((len(session_lengths_mouse),K))
        for j in range(len(session_lengths_mouse)):   
            runProbs = zprobs_mouse[start:start+session_lengths_mouse[j],:] # get state probabilities for session

            trials_in_each_state = np.zeros((K))        
            for k in range(K):
                state_assignment = np.argmax(runProbs,axis=1)
                trials_in_each_state[k] = len(np.where(state_assignment == k)[0])
                    
            prop_time_in_each_state[j,:] = np.round(trials_in_each_state/len(runProbs),5)
            start = start + session_lengths_mouse[j] # beginning index of next session
        
        prop_time_in_each_state_all_mice = np.vstack((prop_time_in_each_state_all_mice,prop_time_in_each_state))
        prop_time_in_each_state_all_mice = prop_time_in_each_state_all_mice[1:,:]

    for i in range(1,len(prop_time_in_each_state_all_mice)):
        color = colors.T*prop_time_in_each_state_all_mice[i,:]
        color[color == 0] = np.nan
        # plot each dot with a small amount of gaussian noise for easier visualization
        ax.plot(prop_time_in_each_state_all_mice[i,0]+np.random.normal(loc=0,scale=0.03),\
                 prop_time_in_each_state_all_mice[i,1]+np.random.normal(loc=0,scale=0.03),\
                    'o',color=np.nanmean(color.T,axis=0), markersize=4)

    ax.set_xlabel('p(state 1)')
    ax.set_ylabel('p(state 2)')

def plot_states_each_session(z,sessions,mouseIDs,ax):

    K = len(np.unique(z)) # number of states
    session_lengths = np.diff(sessions)
    unique_mouse_IDs = np.unique(mouseIDs)

    percent_sessions = [[] for i in range(K)] # initialize empty list for each possible number of states
    for i in unique_mouse_IDs: # for each mouse
        z_mouse = z[mouseIDs==i] # get subset of state probabilities for mouse
        ixs_mouse = np.where(mouseIDs==i)[0] # get trial indices for mouse
        session_num_start = np.where(sessions==ixs_mouse[0])[0][0] # index of first session length for mouse
        session_num_stop = np.where(sessions==ixs_mouse[-1]+1)[0][0]-1 # index of last session length for mouse
        
        session_lengths_mouse = session_lengths[session_num_start:session_num_stop] # list of session lengths for mouse
            
        number_of_states  = [[] for i in range(K)] # initialize empty list for each possible number of states

        for j in range(len(session_lengths_mouse)): # for each session for each mouse
            start = np.sum(session_lengths_mouse[:j])
            stop = start + session_lengths_mouse[j]
            z_mouse_session = z_mouse[start:stop]

            # sort sessions into appropriate lists depending on the number of unique states that appear
            for k in range(K):
                if len(np.unique(z_mouse_session)) == k+1:
                    number_of_states[k].append(j) # append session number to appropriate list

        # compute the percentage of sessions that each number of states equates to
        for k in range(K):
            percent_sessions[k].append(len(number_of_states[k])/len(session_lengths_mouse))

    average_percent_sessions = np.mean(np.array(percent_sessions),axis=1)

    ax.bar([1,2,3],average_percent_sessions,color=[0.7,0.7,0.7])
    ax.plot([1,2,3],percent_sessions,'ko',markersize=2)
    #ax.set_xlabel('# of states')
    ax.set_ylim([0,1])
    ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.set_ylabel('% of sessions')

def plot_state_occupancies(z,mouseIDs,colors,ax):

    K = len(np.unique(z)) # number of states
    unique_mouse_IDs = np.unique(mouseIDs)
    num_mice = len(unique_mouse_IDs)
    percent_time = np.zeros((K,num_mice))

    for i in range(K):
        for j in range(num_mice):
            mouse_ixs = np.where(mouseIDs==unique_mouse_IDs[j])[0] # ID of mouse
            percent_time[i,j] = len(np.where(z[mouse_ixs]==i)[0])/len(z[mouse_ixs])*100
            
    avg_percent_time = np.mean(percent_time, axis=1)
        
    Labels = ['state %s' %(i+1) for i in range(K)]
    ax.bar(Labels,avg_percent_time, color = colors)
    ax.plot(Labels,percent_time,'ko',markersize=2)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(Labels,rotation=90)
    ax.set_yticks([0,30,60,90])
    ax.set_ylabel('time in state (%)')

def plot_simulated_vs_true_transitions(A_true,A_sim,ax,colors=None,diag=True):
    '''
    A_true : kxk matrix containing the values of the true transition probabilities
    A_sim : sxkxk matrix containing the values of the simulated transition probabilities, where s is the number of simulations
    ax : the figure axis handle
    colors : a list of size three arrays containing the RBG color values for plotting the simulated and true transition probabilities
    diag : boolean, default=True. If True, plots the on-diagonal transition probabilities. If False, plots the off-diagonals.
    '''

    num_sims = A_sim.shape[0]
    K = A_true.shape[0]

    if colors is None: 

        colors = [np.array([165,165,165])/255,np.array([0,0,0])]

    if diag: # plot diagonal values

        # plot simulated transition probabilities
        for i in range(num_sims):
            diags_simulated = np.zeros(K)
            for j in range(K):
                diags_simulated[j] = A_sim[i,j,j]
            ax.plot(diags_simulated,'.',color=colors[0],markersize=10)

        # plot true transition probabilities
        diags_true = np.zeros(K)
        for i in range(K):
            diags_true[i] = A_true[i,i]
        ax.plot(diags_true,'.',color=colors[1],markersize=10)

        # format plot
        ax.set_ylabel('$p(z_{t+1} | z_t)$')
        ax.set_xlabel('transitions')
        ax.set_title('diagonal',fontsize=24)
        ax.set_yticks([0.985,0.990,0.995])
        ax.set_ylim([0.9845,0.995])
        ax.set_xticks(np.arange(K))
        xlabels = ["$P_{%s%s}$" %(i+1,i+1) for i in range(K)]
        ax.set_xticklabels(xlabels)
        



    else: # plot off-diagonal values

        # plot simulated transition probabilities
        for i in range(num_sims):
            offdiags_simulated = np.zeros((K*K)-K)
            count = 0
            for j in range(K):
                for k in range(K):
                    if j != k:
                        offdiags_simulated[count] = A_sim[i,j,k]
                        count += 1
            # only add legend once 
            if i == 0:
                ax.plot(offdiags_simulated,'.',color=colors[0],markersize=10, label = 'simulation')
            else: 
                ax.plot(offdiags_simulated,'.',color=colors[0],markersize=10)
        

        # plot true transition probabilities
        offdiags_real = np.zeros((K*K)-K)
        count = 0
        xlabels = []
        for j in range(K):
            for k in range(K):
                if j != k:
                    offdiags_real[count] = A_true[j,k]
                    xlabels.append('$P_{%s%s}$' %(j+1,k+1))
                    count += 1

        ax.plot(offdiags_real,'.',color=colors[1],markersize=10,label= 'data')

        # format plot
        ax.set_ylabel('$p(z_{t+1} | z_t)$')
        ax.set_xlabel('transitions')
        ax.set_title('off-diagonal',fontsize=24)
        ax.set_yticks(np.arange(0,0.021,0.005))
        ax.set_xticks(np.arange((K*K)-K))
        ax.set_xticklabels(xlabels)    
        ax.legend(fontsize=18)    



      