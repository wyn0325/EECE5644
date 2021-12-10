%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EECE5644 Fall 2021
% Wang Yinan 001530926 | HW3
%%=========================Question 2=========================%%
% Code help and example from Prof.Deniz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;close all;clc;

% variance
n=2;
alpha_true=[0.20,0.30,0.23,0.27];
% mu_true=[10 -10 -10 10;-10 10 -10 10];
mu_true(:,1) = [10;-10];
mu_true(:,2) = [-10;10];
mu_true(:,3) = [-10;-10];
mu_true(:,4) = [10;10];
Sigma_true(:,:,1) = [15 1;1 15];
Sigma_true(:,:,2) = [17 3;3 17];
Sigma_true(:,:,3) = [19 5;5 19];
Sigma_true(:,:,4) = [21 7;7 21];

% Number of samples
N=[10,100,1000,10000];

% ensure the program is not stuck
countN = 0;

num_GMM_picks = zeros(length(N),6);
num_GMM_cmp = zeros(length(N),6);

% multi experiments
for a=1:30
    for i=1:length(N)
        [x,label]=generate_samples(N(i),mu_true,Sigma_true,alpha_true);
        GMM_pick=cross_val(x);
        num_GMM_picks(i,GMM_pick)=num_GMM_picks(i,GMM_pick)+1;
    end
    if ~isequal(num_GMM_cmp, num_GMM_picks)
        figure,
        bar(num_GMM_picks');
        legend('10 Training Samples','100 Training Samples', ...
            '1000 Training Sample','10000 Training Sample');
        title('GMM Model Order Selection');
        xlabel('GMM Model Order');ylabel('Frequency of Selection');
        saveas(gcf,['./Q2figs/4-',int2str(a),'.jpg']);
        num_GMM_cmp=num_GMM_picks;
    end
end


for i=1:length(N)
    countN = countN+1
    % Create appropriate number of data points from each distribution
    [x,label]=generate_samples(N(i),mu_true,Sigma_true,alpha_true);
   
    % plot
    figure(i);
    scatter(x(1,label==1),x(2,label==1),'r','filled');
    hold on
    scatter(x(1,label==2),x(2,label==2),'g','filled');
    hold on
    scatter(x(1,label==3),x(2,label==3),'b','filled');
    hold on
    scatter(x(1,label==4),x(2,label==4),'m','filled');
    title(strcat('Data with N=',num2str(N(i))));
    xlabel('x_1'),ylabel('x_2')
    saveas(gcf,['./Q2figs/',int2str(i),'.jpg']);

    GMM_pick=cross_val(x);
    num_GMM_picks(i,GMM_pick)=num_GMM_picks(i,GMM_pick)+1;

    %Tolerance for EM stopping criterion
    delta = 1e-4;
    %Regularization parameter for covariance estimates
    regWeight = 1e-10; 
    %K-Fold Cross Validation
    K = 10; 

    %To determine dimensionality of samples and number of GMM components
    [d,MM] = size(mu_true); 

    %Divide the data set into 10 approximately-equal-sized partitions
    dummy = ceil(linspace(0,N(i),K+1));
    for k = 1:K
        indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
    end
    %Allocate space
    loglikelihoodtrain = zeros(K,6); loglikelihoodvalidate = zeros(K,6); 
    Averagelltrain = zeros(1,6); Averagellvalidate = zeros(1,6);

    countM = 0;
    %Try all 6 mixture options
    for M = 1:6

        countM = countM+1
        countk = 0;

        %10-fold cross validation
        for k = 1:K
            countk = countk+1
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            %Using folk k as validation set
            x1Validate = x(1,indValidate); 
            x2Validate = x(2,indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N(i)];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,2):N(i)];
            end
            
            %Using all other folds as training set
            x1Train = x(1,indTrain); 
            x2Train = x(2,indTrain);
            xTrain = [x1Train; x2Train];
            xValidate = [x1Validate; x2Validate];
            Ntrain = length(indTrain); Nvalidate = length(indValidate);
            
            %Train model parameters (EM)
            %Initialize the GMM to randomly selected samples
            alpha = ones(1,M)/M;
            shuffledIndices = randperm(Ntrain);
            %Pick M random samples as initial mean estimates (this led
            %to good initial estimates (better log likelihoods))
            mu = xTrain(:,shuffledIndices(1:M)); 
            %Assign each sample to the nearest mean (better initialization)
            [~,assignedCentroidLabels] = min(pdist2(mu',xTrain'),[],1); 
            %Use sample covariances of initial assignments as initial covariance estimates
            for m = 1:M 
                Sigma(:,:,m) = cov(xTrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
            end
            t = 0;
            
            %Not converged at the beginning
            Converged = 0; 

            while ~Converged
                for l = 1:M
                    temp(l,:) = repmat(alpha(l),1,Ntrain).*evalGaussian(xTrain,mu(:,l),Sigma(:,:,l));
                end
                plgivenx = temp./sum(temp,1);
                clear temp
                alphaNew = mean(plgivenx,2);
                w = plgivenx./repmat(sum(plgivenx,2),1,Ntrain);
                muNew = xTrain*w';
                for l = 1:M
                    v = xTrain-repmat(muNew(:,l),1,Ntrain);
                    u = repmat(w(l,:),d,1).*v;
                    %Adding a small regularization term
                    SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); 
                end
                Dalpha = sum(abs(alphaNew-alpha));
                Dmu = sum(sum(abs(muNew-mu)));
                DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                %Check if converged
                Converged = ((Dalpha+Dmu+DSigma)<delta); 
                alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
                t = t+1;
            end
            %Validation
            loglikelihoodtrain(k,M) = sum(log(evalGMM(xTrain,alpha,mu,Sigma)));
            loglikelihoodvalidate(k,M) = sum(log(evalGMM(xValidate,alpha,mu,Sigma)));
           
        end
        
        %Average Performance Variables
        Averagelltrain(1,M) = mean(loglikelihoodtrain(:,M)); 
        BICtrain(1,M) = -2*Averagelltrain(1,M)+M*log(N(i));
        Averagellvalidate(1,M) = mean(loglikelihoodvalidate(:,M)); 
        %Sometimes the log likelihoods for N=10 are zero, leading to
        %negative infinity results. I assume that this is instead the
        %lowest log likelihood value instead (so it is possible to graph).
        if isinf(Averagellvalidate(1,M))
            Averagellvalidate(1,M) = (min(Averagellvalidate(find(isfinite(Averagellvalidate)))));
        end
        BICvalidate(1,M) = -2*Averagellvalidate(1,M)+M*log(N(i));
        %Recording values
        TotBICValidate(i,M) = BICvalidate(1,M);
        TotBICTrain(i,M) = BICtrain(1,M);
        TotAvgllValidate(i,M) = Averagellvalidate(1,M);
        TotAvgllTrain(i,M) = Averagelltrain(1,M);
    end
    %Recording Best Outcomes
    [LowestBIC orderB] = min(BICvalidate)
    [Lowestll orderl] = max(Averagellvalidate)

    % training log-likelihood
    figure(i+4), clf,
    plot(Averagelltrain,'.b'); 
    hold on;
    plot(Averagelltrain,'-b'); 
    xlabel('GMM Number'); ylabel(strcat('Log likelihood estimate with ',num2str(K),'-fold cross-validation'));
    title(strcat('Training Log-Likelihoods for N=',num2str(N(i))));
    grid on
    xticks(1:1:6)
    saveas(gcf,['./Q2figs/',int2str(i+4),'.jpg']);
    
    % validation log-likelihood
    figure(i+8), clf,
    plot(Averagellvalidate,'rx');
    hold on;
    plot(Averagellvalidate,'r-');
    xlabel('GMM Number'); ylabel(strcat('Log likelihood estimate with ',num2str(K),'-fold cross-validation'));
    title(strcat('Validation Log-Likelihoods for N=',num2str(N(i))));
    grid on
    xticks(1:1:6)
    saveas(gcf,['./Q2figs/',int2str(i+8),'.jpg']);
    
    % training BIC
    figure(i+12), clf,
    plot(BICtrain,'.b');
    hold on;
    plot(BICtrain,'-b');
    xlabel('GMM Number'); ylabel(strcat('BIC estimate with ',num2str(K),'-fold cross-validation'));
    title(strcat('Training BICs for N=',num2str(N(i))));
    grid on
    xticks(1:1:6)
    saveas(gcf,['./Q2figs/',int2str(i+12),'.jpg']);
    
    % validation BIC
    figure(i+16), clf,
    plot(BICvalidate,'rx');
    hold on;
    plot(BICvalidate,'r-');
    xlabel('GMM Number'); ylabel(strcat('BIC estimate with ',num2str(K),'-fold cross-validation'));
    title(strcat('Validation BICs for N=',num2str(N(i))))
    grid on
    xticks(1:1:6)
    saveas(gcf,['./Q2figs/',int2str(i+16),'.jpg']);
    
    %Saving values
    BICorder(i) = orderB;
    BIClow(i) = LowestBIC;
    lorder(i) = orderl;
    lllow(i) = Lowestll;
end


%%=========================Question 2 Functions=========================%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions credit to Prof.Deniz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
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
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function best_GMM=cross_val(x)
%PerformsEMalgorithmtoestimateparametersandevalueteperformance
%oneachdatasetBtimes,with1throughMGMMmodelsconsidered
B=10;M=6;%repetitionsperdataset;maxGMMconsidered
perf_array=zeros(B,M);%savespaceforperformanceevaluation
%Testeachdataset10times
for b=1:B
    %Pickrandomdatapointstofilltrainingandvalidationsetand
    %addnoise
    set_size=500;
    train_index=randi([1,length(x)],[1,set_size]);
    train_set=x(:,train_index)+(1e-3)*randn(2,set_size);
    val_index=randi([1,length(x)],[1,set_size]);
    val_set=x(:,val_index)+(1e-3)*randn(2,set_size);
    for m=1:M
        %Non􀀀Built􀀀In:runEMalgorithtoestimateparameters
        %[alpha,mu,sigma]=EMforGMM(m,trainset,setsize,valset);
        %Built􀀀Infunction:runEMalgorithmtoestimateparameters
        GMModel=fitgmdist(train_set',M,'RegularizationValue',1e-10);
        alpha=GMModel.ComponentProportion;
        mu=(GMModel.mu)';
        sigma=GMModel.Sigma;
        %Calculatelog􀀀likelihoodperformancewithnewparameters
        perf_array(b,m)=sum(log(evalGMM(val_set,alpha,mu,sigma)));
    end
end
% Ca l cul a t e average per formance f o r each M and f i n d be s t f i t
avg_perf=sum(perf_array)/B;
best_GMM=find(avg_perf==max(avg_perf),1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,label]=generate_samples(N,mu_true,Sigma_true,alpha_true)
% Create appropriate number of data points from each distribution
x=zeros(2,N);
label=zeros(1,N);
for j=1:N
    r=rand(1);
    if r <= alpha_true(1)
        label(j)=1;
    elseif (alpha_true(1)<r)&&(r<=sum(alpha_true(1:2)))
        label(j)=2;
    elseif (sum(alpha_true(1:2))<r)&&(r<=sum(alpha_true(1:3)))
        label(j)=3;
    else
        label(j)=4;
    end
end
Nc=[sum(label==1),sum(label==2),sum(label==3),sum(label==4)];
%{
% when the samples' num is small(like 10)
% there could be non-generated class
if ismember(0,Nc)
    % find non-generated class
    a=find(Nc==0);
    % add 1
    Nc(a)=1;
    % which class's num is the max
    b=find(Nc==max(Nc));
    % minus 1 to keep the total nums
    Nc(b)=Nc(b)-1;
    % find the max-class position in label
    c=find(label==b);
    % change the first position to non-generated class
    label(c(1))=a;
end
%}
% Generate data
x(:,label==1)=randGaussian(Nc(1),mu_true(:,1),Sigma_true(:,:,1));
x(:,label==2)=randGaussian(Nc(2),mu_true(:,2),Sigma_true(:,:,2));
x(:,label==3)=randGaussian(Nc(3),mu_true(:,3),Sigma_true(:,:,3));
x(:,label==4)=randGaussian(Nc(4),mu_true(:,4),Sigma_true(:,:,4));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%