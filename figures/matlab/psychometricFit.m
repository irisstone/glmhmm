function psych = psychometricFit(choice,nCues_RminusL,bins,inits,lb,ub,fitWhat,fitNotBinned,fitBinned)

psych.perfPsychChoiceSize = choice;
psych.perfPsychDiffTSize = nCues_RminusL;
% psych = psychometricFit(choice,nCues_RminusL,fitNotBinned)

psych.perfPsych_bins = bins;
if nargin < 5
    lb = [1e-6 1e-6 -inf -inf];
end
if nargin < 6
    ub = [0.75 0.75  inf inf];
end
if nargin < 7
    fitWhat = 'mle';
end
if nargin < 8
    fitNotBinned = 1;
end
if nargin < 9
    fitBinned = 1;
end

% psychometric curve from raw data (all diffs and binned)
psych.stdInterval         = normcdf(1, 0, 1) - normcdf(-1, 0, 1);
psych.perfPsych_xaxis     = -15:15;%unique(nCues_RminusL);
% psych.perfPsych_xaxis(isnan(psych.perfPsych_xaxis)) = [];
% psych.perfPsych_xaxis(psych.perfPsych_xaxis > 15 | psych.perfPsych_xaxis < -15) = [];

psych.perfPsych           = zeros(size(psych.perfPsych_xaxis));
psych.nR                  = psych.perfPsych;
psych.nTrials             = psych.perfPsych;
psych.perfPsych_xaxisBins = zeros(1,numel(psych.perfPsych_bins)-1);
psych.perfPsych_binned    = zeros(size(psych.perfPsych_xaxisBins));
psych.nR_binned           = psych.perfPsych_binned;
psych.nTrials_binned      = psych.perfPsych_binned;

% all #r - #l, fraction went right
for ii = 1:length(psych.perfPsych_xaxis)
    psych.nR(ii)        = sum(choice(nCues_RminusL==psych.perfPsych_xaxis(ii)) == 1);
    psych.nTrials(ii)   = sum(nCues_RminusL==psych.perfPsych_xaxis(ii));
    psych.perfPsych(ii) = psych.nR(ii)/psych.nTrials(ii); 
end
% binned
for ii = 1:length(psych.perfPsych_bins)-1
    psych.nR_binned(ii)           = sum(choice(nCues_RminusL>=psych.perfPsych_bins(ii) & ...
        nCues_RminusL<psych.perfPsych_bins(ii+1)) == 0);
    psych.nTrials_binned(ii)      = sum(nCues_RminusL>=psych.perfPsych_bins(ii)        & ...
        nCues_RminusL<psych.perfPsych_bins(ii+1));
    psych.perfPsych_binned(ii)    =  psych.nR_binned(ii)/psych.nTrials_binned(ii); 
    
    % bin center needs to be weighted average of delta per ntrials
    dvals = psych.perfPsych_bins(ii):psych.perfPsych_bins(ii+1) -1;
    for jj = 1:numel(dvals); nt(jj) = sum(nCues_RminusL==dvals(jj)); end
    try
      psych.perfPsych_xaxisBins(ii) = sum(dvals.*nt)./sum(nt);
    catch
      psych.perfPsych_xaxisBins(ii) = psych.perfPsych_bins(ii) + mode(diff(psych.perfPsych_bins))/2;
    end
end

% psychometric curve using jeffrey's method (all diffs and binned)
% [psych.perfPsychJ,psych.perfPsychJSEM]               = ...
%     binofit(psych.nR,psych.nTrials,1-psych.stdInterval);
% [psych.perfPsychJ_binned,psych.perfPsychJ_binnedSEM] = ...
%     binofit(psych.nR_binned,psych.nTrials_binned,1-psych.stdInterval);
[psych.perfPsychJ,psych.perfPsychJSEM]               = ...
    binointerval(psych.nR,psych.nTrials,1-psych.stdInterval);
[psych.perfPsychJ_binned,psych.perfPsychJ_binnedSEM] = ...
    binointerval(psych.nR_binned,psych.nTrials_binned,1-psych.stdInterval);

% 4-parameter sigmoid curve
% Sigmoid function fit, weighted by 1/sigma^2 where sigma is the symmetrized error
% psych.sigmoid         = @(O,A,lambda,x0,x) O + A ./ (1 + exp(-(x-x0)/lambda));
% psych.sigmoidSlope    = @(A,lambda) A ./ (4*lambda); % derivitative of the curve at delta 0 towers
sigmoid         = @(O,A,lambda,x0,x) O + A ./ (1 + exp(-(x-x0)/lambda));
sigmoidSlope    = @(A,lambda) A ./ (4*lambda); % derivitative of the curve at delta 0 towers
% psych.sigmoid         = sigmoid(O,A,lambda,x0,x);
% psych.sigmoidSlope    = sigmoidSlope(A,lambda); % derivitative of the curve at delta 0 towers
% O, the minimum ‘P(Went Right)’; 
% O + A is the maximum ‘P(Went Right)’. 
% lambda, the slope of the sigmoid; 
% x0, the inflection point of the sigmoid; 
% x is the click difference on each trial (#Right Clicks ? #Left Clicks), 
% y is ‘P(Went Right)’, 

% fit on binned
if fitBinned
try
psych.fit.xaxis       = psych.perfPsych_bins(1):0.05:psych.perfPsych_bins(end);
[params_binned, gof_binned]      = fit ( psych.perfPsych_xaxisBins(~isnan(psych.perfPsych_binned))', ...
                                           psych.perfPsych_binned(~isnan(psych.perfPsych_binned))',          ...
                                           sigmoid,                                                    ...
                                           'StartPoint', [0 1 8 0],                                          ...
                                           'Weights' , ((psych.perfPsychJ_binnedSEM(~isnan(psych.perfPsych_binned),2)- psych.perfPsychJ_binnedSEM(~isnan(psych.perfPsych_binned),1)) / 2).^-2,  ...
                                           'MaxIter' , 400,                                                  ...
                                           'Lower'      , [-0.5 -2 0  -1 ],                                ...
                                           'Upper'      , [ 0.5  2 100  1 ] );   
psych.fit.stdInt      = predint(params_binned, 0, psych.stdInterval, 'functional');
psych.fit.bias        = params_binned(0);
psych.fit.biasCI      = predint(params_binned, 0, 0.95, 'functional')';
psych.fit.biasErr     = (psych.fit.stdInt(2) - psych.fit.stdInt(1)) / 2;
psych.fit.slope       = sigmoidSlope(params_binned.A, params_binned.lambda);
psych.fit.a           = params_binned.A; 
psych.fit.lambda      = params_binned.lambda;
psych.fit.curve       = feval(params_binned,psych.fit.xaxis');
psych.fit.O           = params_binned.O;
psych.fit.xNaught     = params_binned.x0;
catch
    psych.fit = [];
end
end

% fit on full
if fitNotBinned
%try
psych.fitAll.xaxis       = psych.perfPsych_xaxis(1):0.05:psych.perfPsych_xaxis(end);
switch fitWhat
    case 'ls'
[params_all, gof_all]  = fit ( psych.perfPsych_xaxis(~isnan(psych.perfPsychJ))',     ...
                                     psych.perfPsychJ(~isnan(psych.perfPsychJ)),                 ...
                                     sigmoid,                                              ...
                                     'StartPoint' , [0 1 8 0],                                   ...
                                     'Weights'    , ((psych.perfPsychJSEM(~isnan(psych.perfPsychJ),2)- psych.perfPsychJSEM(~isnan(psych.perfPsychJ),1)) / 2).^-2  , ...
                                     'MaxIter'    , 400 ,                                        ...
                                     'Lower'      , [-0.5 -2 0  -20 ],                           ...
                                     'Upper'      , [ 0.5  2 20  20 ] ); 
psych.fitAll.stdInt      = predint(params_all, 0, psych.stdInterval, 'functional');
psych.fitAll.bias        = params_all(0);
psych.fitAll.biasCI      = predint(params_all, 0, 0.95, 'functional')';
psych.fitAll.biasErr     = (psych.fitAll.stdInt(2) - psych.fitAll.stdInt(1)) / 2;
psych.fitAll.slope       = sigmoidSlope(params_all.A, params_all.lambda);
psych.fitAll.a           = params_all.A; 
psych.fitAll.lambda      = params_all.lambda;
psych.fitAll.curve       = feval(params_all,psych.fitAll.xaxis');
psych.fitAll.O           = params_all.O;
psych.fitAll.xNaught     = params_all.x0;
    case 'mle'
params_all      = psychFitMLE(nCues_RminusL,choice,inits,lb,ub);
%psych.fitAll.stdInt      = predint(params_all, 0, psych.stdInterval, 'functional');
%psych.fitAll.bias        = params_all(0);
%psych.fitAll.biasCI      = predint(params_all, 0, 0.95, 'functional')';
psych.fitAll.negL        =  params_all.negL;
psych.fitAll.slope       = sigmoidSlope(params_all.A, params_all.lambda);
psych.fitAll.a           = params_all.A; 
psych.fitAll.lambda      = params_all.lambda;
psych.fitAll.curve       = sigmoid(params_all.O,params_all.A,params_all.lambda,params_all.x0,psych.fitAll.xaxis');
psych.fitAll.O           = params_all.O;
psych.fitAll.xNaught     = params_all.x0;
psych.fitAll.pstore      = params_all.pstore;
end
%catch
%    psych.fitAll = [];
end
end
