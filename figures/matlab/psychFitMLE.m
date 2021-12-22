function handle = psychFitMLE(diffT,choice,ninit,lb,ub)
% MLE fitting function with inputs: 
    % xdat: nx1 vector
    % ydat: nx1 vector of binary 0/1 data
    % lb: 1x4 vector of lower bounds for parameters
    % ub: 1x4 vector of upper bounds for parameters
    % ninit: integer, number of random initializations to try
 if nargin < 3
    ninit = 100;
end   

if nargin < 4
    lb = [1e-6 1e-6 -inf -inf];
end
if nargin < 5
    ub = [0.75 0.75  inf inf];
end

% define the psychometric function
% p1 = right lapse rate
% p2 = left lapse rate
% p3 = offset
% p4 = slope
%sigmoid = @(p,x)(p(1)+(1-p(1)-p(2))./(1 + exp(-(x-p(3))*p(4))));

y = ~choice'; % flip 0/1 coding so left/right

% sort arrays along x-axis
[xdat, x_order] = sort(diffT);
xdat = xdat';
ydat = y(x_order,:);

% now fit it
lfun = @(prs)(neglogliSigmoid(prs,xdat,ydat));

% try out a bunch of random inititializations and pick the best one
p1rnd = rand(ninit,1)*0.4;
p2rnd = rand(ninit,1)*0.4;
p3rnd = randn(ninit,1)*0.5;
p4rnd = randn(ninit,1)*0.5;

pp0 = [p1rnd, p2rnd, p3rnd, p4rnd]';
opts = optimset('display', 'off');
neglstore = zeros(ninit,1);
pstore = zeros(4,ninit);
for jj = 1:ninit
    [pstore(:,jj),neglstore(jj)] = fmincon(lfun,pp0(:,jj),[],[],[],[],lb,ub,[],opts);
    %fprintf('iter %d:  negl=%.3f\n', jj,neglstore(jj));
end

[~,jjmin] = min(neglstore);

handle.O = pstore(1,jjmin);
handle.A = 1 - pstore(1,jjmin) - pstore(2,jjmin);
handle.x0 = pstore(3,jjmin);
handle.lambda = 1/pstore(4,jjmin);
handle.negL = min(neglstore);
handle.pstore = pstore;

disp(min(neglstore));

%%
