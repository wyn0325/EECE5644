%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EECE5644 Fall 2021
% Wang Yinan 001530926 | HW4
%%=========================Question 2=========================%%
% Code help and example from Prof.Deniz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all;
%%=========================Setup=========================%%
file = ["135069.jpg"];
K = 10;
M = 10;
n = size(file, 1);

%%=========================Segmentation=========================%%
for i=1:length(file)
    imdata  = imread(file(i));
    figure(1), subplot(1, 2, 1*2-1),
    imshow(imdata);
    title("shows the original photo"); hold on;
    [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);
    rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
    features = [rowIndices(:)';colIndices(:)']; % initialize with row and column indices
    for d = 1:D
        imdatad = imdata(:,:,d); % pick one color at a time
        features = [features;imdatad(:)'];
    end
    minf = min(features,[],2); maxf = max(features,[],2);
    ranges = maxf-minf;
    x = diag(ranges.^(-1))*(features-repmat(minf,1,N));
    d = size(x,1);
    model = 2;
    gm = fitgmdist(x',model);
    p = posterior(gm, x');
    [~, l] = max(p,[], 2);
    li = reshape(l, R, C);
    figure(1), subplot(n, 2, 1*2)
    imshow(uint8(li*255/model));
    title(strcat("Clustering with K=", num2str(model)));
    ab = zeros(1,M);
    for model = 1:M
        ab(1,model) = calcLikelihood(x, model, K);
    end
    [~, mini] = min(ab);
    gm = fitgmdist(x', mini);
    p = posterior(gm, x');
    [~, l] = max(p,[], 2);
    li = reshape(l, R, C);
    figure(2), subplot(n,1,1),
    imshow(uint8(li*255/mini));
    title(strcat("Best Clustering with K=", num2str(mini)));
    fig=figure(3); 
    subplot(1,n,1), plot(ab,'-b');
end

rst = axes(fig, 'visible', 'off');
rst.Title.Visible='on';
rst.XLabel.Visible='on';
rst.YLabel.Visible='on';
ylabel(rst,'Negative Loglikelihood');
xlabel(rst,'Model Order');
title(rst,['Negative Loglikelihood']);

%%=========================Question2 functions=========================%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function negativeLoglikelihood = calcLikelihood(x, model, K)
    N = size(x,2);
    dummy = ceil(linspace(0, N, K+1));
    negativeLoglikelihood = 0;
    for k=1:K
        indPartitionLimits(k,:) = [dummy(k) + 1, dummy(k+1)];
    end
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xv = x(:, indValidate); % Using folk k as validation set
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
        end
        xt = x(:, indTrain);
        try
            gm = fitgmdist(xt', model);
            [~, nlogl] = posterior(gm, xv');
            negativeLoglikelihood = negativeLoglikelihood + nlogl;
        catch exception
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%