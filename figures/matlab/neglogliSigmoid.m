function negL = neglogliSigmoid(prs,xdat,ydat)
% negL = neglogliSigmoid(prs,xdat,ydat)
%
% Computes negative log-likelihood of data (xdat,ydat) under sigmoid
% function with left and right lapse rate
%
% Inputs:
%  prs [4 x 1] = [right-lapse; left-lapse; offset; slope]
% xdat [n x 1] = x coordinates
% ydat [n x 1] = binary (0/1) values

% Unpack the parameters
rlapse = prs(1);
llapse = prs(2);
offset = prs(3);
slope = prs(4);

% Compute Bernoulli probabilities
pvals = rlapse + (1-rlapse-llapse)./(1 + exp(-(xdat-offset)*slope));

% check to make sure we didn't get neg vals or vals > 1
if any((pvals<1e-14) | (pvals>1-1e-14))
    iilow = (pvals<1e-14);
    iihi = (pvals>1-1e-14);
    pvals(iilow) = 1e-14;
    pvals(iihi) = 1-1e-14;
else
    iilow=[]; iihi=[];
end

% compute neg log li
negL = - ydat'*log(pvals) - (1-ydat)'*log(1-pvals);

% add a penalty for params outside range (to help avoid this range)
if ~isempty(iilow)
    negL = negL+sum(iilow)*1e6;
end    

if ~isempty(iihi)
    negL = negL+sum(iihi)*1e6;
end    
