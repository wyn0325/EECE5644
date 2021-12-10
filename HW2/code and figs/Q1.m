%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EECE5644 Fall 2021
% Wang Yinan 001530926 | HW2
%%=========================Question 1=========================%%
% Code help and example from Prof.Deniz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;close all;clc;

%%=========================Setup=========================%%
dimension=2; %Dimension of data
%Define data
D.d100.N=100;
D.d1000.N=1000;
D.d10k.N=10000;
D.d20k.N=20e3;
dTypes=fieldnames(D);
%Define Statistics
p=[0.6 0.4]; %Prior
%Label 0 GMM Stats
mu0=[5 0;0 4]';
Sigma0(:,:,1)=[4 0;0 2];
Sigma0(:,:,2)=[1 0;0 3];
alpha0=[0.5 0.5];
%Label 1 Single Gaussian Stats
mu1=[3 2]';
Sigma1=[2 0;0 2];
alpha1=1;
figure(1);
%Generate Data
for ind=1:length(dTypes)
    D.(dTypes{ind}).x=zeros(dimension,D.(dTypes{ind}).N); %Initialize Data
    %Determine Posteriors
    D.(dTypes{ind}).labels = rand(1,D.(dTypes{ind}).N)>=p(1);
    D.(dTypes{ind}).N0=sum(~D.(dTypes{ind}).labels);
    D.(dTypes{ind}).N1=sum(D.(dTypes{ind}).labels);
    D.(dTypes{ind}).phat(1)=D.(dTypes{ind}).N0/D.(dTypes{ind}).N;
    D.(dTypes{ind}).phat(2)=D.(dTypes{ind}).N1/D.(dTypes{ind}).N;

    [D.(dTypes{ind}).x(:,~D.(dTypes{ind}).labels),...
        D.(dTypes{ind}).dist(:,~D.(dTypes{ind}).labels)]=...
        randGMM(D.(dTypes{ind}).N0,alpha0,mu0,Sigma0);
    [D.(dTypes{ind}).x(:,D.(dTypes{ind}).labels),...
        D.(dTypes{ind}).dist(:,D.(dTypes{ind}).labels)]=...
        randGMM(D.(dTypes{ind}).N1,alpha1,mu1,Sigma1);
    subplot(2,2,ind);
    plot(D.(dTypes{ind}).x(1,~D.(dTypes{ind}).labels),...
        D.(dTypes{ind}).x(2,~D.(dTypes{ind}).labels),'b.','DisplayName','Class 0');
    hold all;
    plot(D.(dTypes{ind}).x(1,D.(dTypes{ind}).labels),...
        D.(dTypes{ind}).x(2,D.(dTypes{ind}).labels),'r.','DisplayName','Class 1');
    grid on;
    xlabel('x1');ylabel('x2');
    title([num2str(D.(dTypes{ind}).N) ' Samples From Two Classes']);
end
legend 'show';

%%=========================Part 1=========================%%
px0=evalGMM(D.d20k.x,alpha0,mu0,Sigma0);
px1=evalGaussian(D.d20k.x ,mu1,Sigma1);
discScore=log(px1./px0);
sortDS=sort(discScore);
%Generate vector of gammas for parametric sweep
logGamma=[min(discScore)-eps sort(discScore)+eps];
prob=CalcProb(discScore,logGamma,D.d20k.labels,D.d20k.N0,D.d20k.N1,D.d20k.phat);
logGamma_ideal=log(p(1)/p(2));
decision_ideal=discScore>logGamma_ideal;
p10_ideal=sum(decision_ideal==1 & D.d20k.labels==0)/D.d20k.N0;
p11_ideal=sum(decision_ideal==1 & D.d20k.labels==1)/D.d20k.N1;
pFE_ideal=(p10_ideal*D.d20k.N0+(1-p11_ideal)*D.d20k.N1)/(D.d20k.N0+D.d20k.N1);
%Estimate Minimum Error
%If multiple minimums are found choose the one closest to the theoretical
%minimum
[prob.min_pFE, prob.min_pFE_ind]=min(prob.pFE);
if length(prob.min_pFE_ind)>1
    [~,minDistTheory_ind]=min(abs(logGamma(prob.min_pFE_ind)-logGamma_ideal));
    prob.min_pFE_ind=prob.min_pFE_ind(minDistTheory_ind);
end
%Find minimum gamma and corresponding false and true positive rates
minGAMMA=exp(logGamma(prob.min_pFE_ind));
prob.min_FP=prob.p10(prob.min_pFE_ind);
prob.min_TP=prob.p11(prob.min_pFE_ind);
%Plot
plotROC(prob.p10,prob.p11,prob.min_FP,prob.min_TP,p10_ideal,p11_ideal);
plotMinPFE(logGamma,prob.pFE,prob.min_pFE_ind);
plotDecisions(D.d20k.x,D.d20k.labels,decision_ideal);
plotERMContours(D.d20k.x,alpha0,mu0,Sigma0,mu1,Sigma1,logGamma_ideal);
fprintf('Theoretical: Gamma=%1.2f, Error=%1.2f%%\n',...
    exp(logGamma_ideal),100*pFE_ideal);
fprintf('Estimated: Gamma=%1.2f, Error=%1.2f%%\n',minGAMMA,100*prob.min_pFE);

%%=========================Part 2=========================%%
roc=zeros(4,20001,3);
samples=[100 1000 10000 20000];
for ind=1:length(dTypes)-1
    %Estimate Parameters using matlab built in function
    D.(dTypes{ind}).DMM_Est0=...
        fitgmdist(D.(dTypes{ind}).x(:,~D.(dTypes{ind}).labels)',2,'Replicates',10);
    D.(dTypes{ind}).DMM_Est1=...
        fitgmdist(D.(dTypes{ind}).x(:,D.(dTypes{ind}).labels)',1);
    plotContours(D.(dTypes{ind}).x,...
        D.(dTypes{ind}).DMM_Est0.ComponentProportion,...
        D.(dTypes{ind}).DMM_Est0.mu,D.(dTypes{ind}).DMM_Est0.Sigma,dTypes{ind});
    %Calculate discriminate score
    px0=pdf(D.(dTypes{ind}).DMM_Est0,D.d20k.x');
    px1=pdf(D.(dTypes{ind}).DMM_Est1,D.d20k.x');
    discScore=log(px1'./px0');
    sortDS=sort(discScore);
    %Generate vector of gammas for parametric sweep
    logGamma=[min(discScore)-eps sort(discScore)+eps];
    prob=CalcProb(discScore,logGamma,D.d20k.labels,...
        D.d20k.N0,D.d20k.N1,D.(dTypes{ind}).phat);
    %Estimate Minimum Error
    %If multiple minimums are found choose the one closest to the theoretical
    %minimum
    [prob.min_pFE, prob.min_pFE_ind]=min(prob.pFE);
    if length(prob.min_pFE_ind)>1
        [~,minDistTheory_ind]=...
            min(abs(logGamma(prob.min_pFE_ind)-logGamma_ideal));
        prob.min_pFE_ind=min_pFE_ind(minDistTheory_ind);
    end
    %Find minimum gamma and corresponding false and true positive rates
    minGAMMA=exp(logGamma(prob.min_pFE_ind));
    prob.min_FP=prob.p10(prob.min_pFE_ind);
    prob.min_TP=prob.p11(prob.min_pFE_ind);
    %Plot
    %plotMinPFE(logGamma,prob.pFE,prob.min_pFE_ind);
    fprintf('Estimated: Gamma=%1.2f, Error=%1.2f%%\n',...
        minGAMMA,100*prob.min_pFE);     
    roc(1,:,ind)=prob.p10;
    roc(2,:,ind)=prob.p11;
    roc(3,:,ind)=prob.min_FP;
    roc(4,:,ind)=prob.min_TP;
end
figure;
for ind=1:length(dTypes)-1
    nameR=('ROC Curve for '+string(samples(ind))+' Samples');
    nameM=('Min.Errror for '+string(samples(ind))+' Samples');
    plot(roc(1,:,ind),roc(2,:,ind),'DisplayName',nameR,'LineWidth',2);
    hold on;
    plot(roc(3,:,ind),roc(4,:,ind),'o','DisplayName',nameM,'LineWidth',2);
    hold on;
end
xlabel('Prob. False Positive');
ylabel('Prob. True Positive');
title('Mininimum Expected Risk ROC Curves for Training Data');
legend 'show';
grid on; box on;

%%=========================Part 3=========================%%
options=optimset('MaxFunEvals',3000,'MaxIter',10000);
for ind=1:length(dTypes)-1
    lin.x=[ones(1,D.(dTypes{ind}).N); D.(dTypes{ind}).x];
    lin.init=zeros(dimension+1,1);
    % [lin.theta,lin.cost]=thetaEst(lin.x,lin.init,D.(dTypes{ind}).labels);
    [lin.theta,lin.cost]=...
        fminsearch(@(theta)(costFun(theta,lin.x,D.(dTypes{ind}).labels)),...
        lin.init,options);
    lin.discScore=lin.theta'*[ones(1,D.d20k.N); D.d20k.x];
    gamma=0;
    lin.prob=CalcProb(lin.discScore,gamma,D.d20k.labels,...
        D.d20k.N0,D.d20k.N1,D.d20k.phat);
    % quad.decision=[ones(D.d20k.N,1) D.d20k.x]*quad.theta>0;
    plotDecisions(D.d20k.x,D.d20k.labels,lin.prob.decisions);
    title(sprintf(['Data and Classifier Decisions Against True Label ' ...
        'for Linear Logistic Fit\nProbability of Error=%1.1f%% ' ...
        'with %s samples'], ...
        100*lin.prob.pFE,string(samples(ind))));
    quad.x=[ones(1,D.(dTypes{ind}).N); D.(dTypes{ind}).x;...
        D.(dTypes{ind}).x(1,:).^2;...
        D.(dTypes{ind}).x(1,:).*D.(dTypes{ind}).x(2,:);...
        D.(dTypes{ind}).x(2,:).^2];
    quad.init= zeros(2*(dimension+1),1);
    [quad.theta,quad.cost]=...
        fminsearch(@(theta)(costFun(theta,quad.x,D.(dTypes{ind}).labels)),...
        quad.init,options);
    quad.xScore=[ones(1,D.d20k.N); D.d20k.x; D.d20k.x(1,:).^2;...
        D.d20k.x(1,:).*D.d20k.x(2,:); D.d20k.x(2,:).^2];
    quad.discScore=quad.theta'*quad.xScore;
    gamma=0;
    quad.prob=CalcProb(quad.discScore,gamma,D.d20k.labels,...
        D.d20k.N0,D.d20k.N1,D.d20k.phat);
    plotDecisions(D.d20k.x,D.d20k.labels,quad.prob.decisions);
    title(sprintf(['Data and Classifier Decisions Against True Label ' ...
        'for Quadratic Logistic Fit\nProbability of Error=%1.1f%% ' ...
        'with %d Samples'], ...
        100*quad.prob.pFE,samples(ind)));
end

%%=========================Question 1 Functions=========================%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions credit to Prof.Deniz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cost=costFun(theta,x,labels)
h=1./(1+exp(-x'*theta));
cost=-1/length(h)*sum((labels'.*log(h)+(1-labels)'.*(log(1-h))));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,labels] = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1));
x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
labels(ind)=m-1;
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z = randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prob=CalcProb(discScore,logGamma,labels,N0,N1,phat)
for ind=1:length(logGamma)
prob.decisions=discScore>=logGamma(ind);
Num_pos(ind)=sum(prob.decisions);
prob.p10(ind)=sum(prob.decisions==1 & labels==0)/N0;
prob.p11(ind)=sum(prob.decisions==1 & labels==1)/N1;
prob.p01(ind)=sum(prob.decisions==0 & labels==1)/N1;
prob.p00(ind)=sum(prob.decisions==0 & labels==0)/N0;
prob.pFE(ind)=prob.p10(ind)*phat(1) + prob.p01(ind)*phat(2);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotContours(x,alpha,mu,Sigma,data)
figure
if size(x,1)==2
plot(x(1,:),x(2,:),'b.');
xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours for ',data),
axis equal, hold on;
rangex1 = [min(x(1,:)),max(x(1,:))];
rangex2 = [min(x(2,:)),max(x(2,:))];
[x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
contour(x1Grid,x2Grid,zGMM); axis equal,
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotROC(p10,p11,min_FP,min_TP,p10_ideal,p11_ideal)
figure;
plot(p10,p11,'DisplayName','ROC Curve','LineWidth',2);
hold all;
plot(min_FP,min_TP,'o','DisplayName','Estimated Min. Error','LineWidth',2);
hold all;
plot(p10_ideal,p11_ideal,'+','DisplayName','Ideal Min. Error');
xlabel('Prob. False Positive');
ylabel('Prob. True Positive');
title('Mininimum Expected Risk ROC Curve');
legend 'show';
grid on; box on;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotMinPFE(logGamma,pFE,min_pFE_ind)
figure;
plot(logGamma,pFE,'DisplayName','Errors','LineWidth',2);
hold on;
plot(logGamma(min_pFE_ind),pFE(min_pFE_ind),...
'ro','DisplayName','Minimum Error','LineWidth',2);
%plot(min_FP,min_TP,'+','DisplayName','Calculated Min. Error');
xlabel('Gamma');
ylabel('Proportion of Errors');
title('Probability of Error vs. Gamma')
grid on;
legend 'show';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotDecisions(x,labels,decisions)
ind00 = find(decisions==0 & labels==0);
ind10 = find(decisions==1 & labels==0);
ind01 = find(decisions==0 & labels==1);
ind11 = find(decisions==1 & labels==1);
figure; % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og','DisplayName','Class 0, Correct'); hold on,
plot(x(1,ind10),x(2,ind10),'or','DisplayName','Class 0, Incorrect'); hold on,
plot(x(1,ind01),x(2,ind01),'+r','DisplayName','Class 1, Correct'); hold on,
plot(x(1,ind11),x(2,ind11),'+g','DisplayName','Class 1, Incorrect'); hold on,
axis equal,
grid on;
title('Data and their classifier decisions versus true labels');
xlabel('x_1'), ylabel('x_2');
legend('Correct decisions for data from Class 0',...
'Wrong decisions for data from Class 0',...
'Wrong decisions for data from Class 1',...
'Correct decisions for data from Class 1');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotERMContours(x,alpha0,mu0,Sigma0,mu1,Sigma1,logGamma_ideal)
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues =...
log(evalGaussian([h(:)';v(:)'],mu1,Sigma1))-log(evalGMM([h(:)';v(:)'],...
alpha0,mu0,Sigma0)) - logGamma_ideal;
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
contour(horizontalGrid,verticalGrid,...
discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0',...
'Wrong decisions for data from Class 0',...
'Wrong decisions for data from Class 1',...
'Correct decisions for data from Class 1',...
'Equilevel contours of the discriminant function' ),
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%