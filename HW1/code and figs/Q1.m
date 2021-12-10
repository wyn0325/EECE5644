%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EECE5644 Fall 2021
% Wang Yinan 001530926 | HW1
%%=========================Question 1=========================%%
% Code help and example from Prof.Deniz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;close all;clc;

%%=========================Setup=========================%%
n=2; %dimensions
N=10000; %samples

% Label 0 GMM Stats
mu0(:,1) = [3;0];
mu0(:,2) = [0;3];
Sigma0(:,:,1)=[2 0;0 1];
Sigma0(:,:,2)=[1 0;0 2];
alpha0=[0.5 0.5];

% Label 1 Single Gaussian Stats
mu1=[2 2]';
Sigma1=[1 0;0 1];
alpha1=1;

% Determine posteriors
p=[0.65,0.35];

% Create appropriate number of data points from each distribution
x=zeros(n,N);
label=rand(1, N) >= p(1);
Nc=[sum(label==0),sum(label==1)];

% Generate data as prescribed in assignment description
x(:,label==0)=randGMM(Nc(1),alpha0,mu0,Sigma0);
x(:,label==1)=randGMM(Nc(2),alpha1,mu1,Sigma1);

% Plot true class labels
figure(1);
plot(x(1,label==0),x(2,label==0),'o',x(1,label==1),x(2,label==1),'+');
title('Class 0 and Class 1 True Class Labels')
xlabel('x_1'),ylabel('x_2')
legend('Class 0','Class 1')


%%=========================Part A=========================%%
% ERM Classification with True Knowledge
px0=evalGMM(x,alpha0,mu0,Sigma0);
px1=evalGaussian(x,mu1,Sigma1);
discScore=log(px1./px0);
sortDS=sort(discScore);

% Generate vector of gammas for parametric sweep
logGamma=[min(discScore)-eps sort(discScore)+eps];
for ind=1:length(logGamma)
    decision=discScore>logGamma(ind);
    Num_pos(ind)=sum(decision);
    pFP(ind)=sum(decision==1 & label==0)/Nc(1);
    pTP(ind)=sum(decision==1 & label==1)/Nc(2);
    pFN(ind)=sum(decision==0 & label==1)/Nc(1);
    pTN(ind)=sum(decision==0 & label==0)/Nc(2);
    %Two ways to make sure I did it right
    pFE(ind)=(sum(decision==0 & label==1) + sum(decision==1 & label==0))/N;
    pFE2(ind)=(pFP(ind)*Nc(1) + pFN(ind)*Nc(2))/N;
end

% Calculate Theoretical Minimum Error
logGamma_ideal=log(p(1)/p(2));
decision_ideal=discScore>logGamma_ideal;
pFP_ideal=sum(decision_ideal==1 & label==0)/Nc(1);
pTP_ideal=sum(decision_ideal==1 & label==1)/Nc(2);
pFE_ideal=(pFP_ideal*Nc(1)+(1-pTP_ideal)*Nc(2))/(Nc(1)+Nc(2));

% Estimate Minimum Error
% If multiple minimums are found choose the one closest to the theoretical
% minimum
[min_pFE, min_pFE_ind]=min(pFE);
if length(min_pFE_ind)>1
    [~,minDistTheory_ind]=min(abs(logGamma(min_pFE_ind)-logGamma_ideal));
    min_pFE_ind=min_pFE_ind(minDistTheory_ind);
end

% Find minimum gamma and corresponding false and true positive rates
minGAMMA=exp(logGamma(min_pFE_ind));
min_FP=pFP(min_pFE_ind);
min_TP=pTP(min_pFE_ind);

% print results
fprintf('Theoretical: Gamma=%1.2f, Error=%1.2f%%\n',...
    exp(logGamma_ideal),100*pFE_ideal);
fprintf('Estimated: Gamma=%1.2f, Error=%1.2f%%\n',minGAMMA,100*min_pFE);

% Plot ROC
figure(2);
plot(pFP,pTP,'DisplayName','ROC Curve','LineWidth',2);
hold all;
plot(min_FP,min_TP,'o','DisplayName','Estimated Min. Error','LineWidth',2);
plot(pFP_ideal,pTP_ideal,'+','DisplayName',...
    'Theoretical Min. Error','LineWidth',2);
xlabel('Prob. False Positive');
ylabel('Prob. True Positive');
title('Mininimum Expected Risk ROC Curve');
legend 'show';
grid on; box on;

% Plot Gamma
figure(3);
plot(logGamma,pFE,'DisplayName','Errors','LineWidth',2);
hold on;
plot(logGamma(min_pFE_ind),pFE(min_pFE_ind),...
    'ro','DisplayName','Minimum Error','LineWidth',2);
xlabel('Gamma');
ylabel('Proportion of Errors');
title('Probability of Error vs. Gamma')
grid on;
legend 'show';


%%=========================Part B=========================%%
% Fisher LDA
% Compute scatter matrices
x0=x(:,label==0)';
x1=x(:,label==1)';
mu0_hat=mean(x0);
mu1_hat=mean(x1);
Sigma0_hat=cov(x0);
Sigma1_hat=cov(x1);

% Compute scatter matrices
Sb=(mu0_hat-mu1_hat)*(mu0_hat-mu1_hat)';
Sw=Sigma0_hat+Sigma1_hat;

% Eigen decompostion to generate WLDA
[V,D]=eig(inv(Sw)*Sb);
[~,ind]=max(diag(D));
w=V(:,ind);
y=w'*x;
w=sign(mean(y(find(label==1))-mean(y(find(label==0)))))*w;
y=sign(mean(y(find(label==1))-mean(y(find(label==0)))))*y;

% Evaluate for different taus
tau=[min(y)-0.1 sort(y)+0.1];
for ind=1:length(tau)
    decision=y>tau(ind);
    Num_pos_LDA(ind)=sum(decision);
    pFP_LDA(ind)=sum(decision==1 & label==0)/Nc(1);
    pTP_LDA(ind)=sum(decision==1 & label==1)/Nc(2);
    pFN_LDA(ind)=sum(decision==0 & label==1)/Nc(2);
    pTN_LDA(ind)=sum(decision==0 & label==0)/Nc(1);
    pFE_LDA(ind)=(sum(decision==0 & label==1)...
        + sum(decision==1 & label==0))/(Nc(1)+Nc(2));
end

% Estimated Minimum Error
[min_pFE_LDA, min_pFE_ind_LDA]=min(pFE_LDA);
minTAU_LDA=tau(min_pFE_ind_LDA);
min_FP_LDA=pFP_LDA(min_pFE_ind_LDA);
min_TP_LDA=pTP_LDA(min_pFE_ind_LDA);

% print results
fprintf('Estimated for LDA: Tau=%1.2f, Error=%1.2f%%\n',...
    minTAU_LDA,100*min_pFE_LDA);

% Plot Fisher LDA Projection
figure(4);
plot(y(label==0),zeros(1,Nc(1)),'o','DisplayName','Label 0');
hold all;
plot(y(label==1),ones(1,Nc(2)),'o','DisplayName','Label 1');
ylim([-1 2]);
plot(repmat(tau(min_pFE_ind_LDA),1,2),ylim,'m--',...
    'DisplayName','Tau for Min. Error','LineWidth',2);
grid on;
xlabel('y');
title('Fisher LDA Projection of Data');
legend 'show';

% Plot ROC
figure(5);
plot(pFP_LDA,pTP_LDA,'DisplayName','ROC Curve','LineWidth',2);
hold all;
plot(min_FP_LDA,min_TP_LDA,'o','DisplayName',...
    'Estimated Min. Error','LineWidth',2);
xlabel('Prob. False Positive');
ylabel('Prob. True Positive');
title('Mininimum Expected Risk ROC Curve');
legend 'show';
grid on; box on;

% Plot Gamma
figure(6);
plot(tau,pFE_LDA,'DisplayName','Errors','LineWidth',2);
hold on;
plot(tau(min_pFE_ind_LDA),pFE_LDA(min_pFE_ind_LDA),'ro',...
    'DisplayName','Minimum Error','LineWidth',2);
xlabel('Tau');
ylabel('Proportion of Errors');
title('Probability of Error vs. Tau for Fisher LDA')
grid on;
legend 'show';

%%=========================Question 1 Functions=========================%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions credit to Prof.Deniz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g=evalGaussian(x,mu,Sigma)
%Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N]=size(x);

C=((2*pi)^n*det(Sigma))^(-1/2);%coefficient
E=-0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);%exponent
g=C*exp(E);%final gaussian evaluationend
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,labels] = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % nality of samples
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