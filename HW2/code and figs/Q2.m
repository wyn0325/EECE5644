%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%EECE5644 Fall 2021
% Wang Yinan 001530926 | HW2
%%=========================Question 2=========================%%
% Code help and example from Prof.Deniz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;close all;clc;

%%=========================Setup=========================%%
% variances
xT = 0.5;
yT = -0.5;
xi = [1, -1, 0, 0];
yi = [0, 0, 1, -1];
SIGMA_X = 0.25;
SIGMA_Y = 0.25;
SIGMA_I = 0.3;
ni = normrnd(0, SIGMA_I^2, 4, 1);
for i = 1:4
    ri(i) = sqrt((xT-xi(i))^2+(yT-yi(i))^2)+ni(i);
end

% Creates an equilevel contour for the MAP estimation objective function
x = linspace(-2,2);
y = linspace(-2,2);
[X,Y] = meshgrid(x,y);
r2 = zeros(10000,4);
s2 = zeros(10000,4);
gMAP = zeros(10000,4);

% Calculates values of the MAP estimation objective function on a given mesh
for i = 1:4
s1 = X(:).^2/(SIGMA_X)^2+Y(:).^2/(SIGMA_Y)^2;
r2(:,i) = sqrt((X(:)-xi(i)).^2+(Y(:)-yi(i)).^2);
s2(:,i) = ((ri(i)-r2(:,i)).^2)/(SIGMA_I^2);
end

% The MAP objective contours
gMAP(:,1) = s1+s2(:,1);
gMAP(:,2) = s1+s2(:,1)+s2(:,2);
gMAP(:,3) = s1+s2(:,1)+s2(:,2)+s2(:,3);
gMAP(:,4) = s1+s2(:,1)+s2(:,2)+s2(:,3)+s2(:,4);
GMAP1 = reshape(gMAP(:,1),100,100);
GMAP2 = reshape(gMAP(:,2),100,100);
GMAP3 = reshape(gMAP(:,3),100,100);
GMAP4 = reshape(gMAP(:,4),100,100);

% PLOT
% for K=1
figure(15);
plot(xi(1),yi(1),'or'); hold on,
plot(xT,yT,'+r'); hold on,
contour(X,Y,GMAP1,'ShowText','on');
legend('landmark location of the object',' true location of the object','MAP objective function contours'), 
title('the MAP objective function contours for K = 1'),
xlabel('x'), ylabel('y')

% for K=2
figure(16);
plot(xi(1:2),yi(1:2),'or'); hold on,
plot(xT,yT,'+r'); hold on,
contour(X,Y,GMAP2,'ShowText','on');
legend('landmark location of the object',' true location of the object','MAP objective function contours'), 
title('the MAP objective function contours for K = 2'),
xlabel('x'), ylabel('y')

% for K=3
figure(17);
plot(xi(1:3),yi(1:3),'or'); hold on,
plot(xT,yT,'+r'); hold on,
contour(X,Y,GMAP3,'ShowText','on');
legend('landmark location of the object',' true location of the object','MAP objective function contours'), 
title('the MAP objective function contours for K = 3'),
xlabel('x'), ylabel('y')

% for K=4
figure(18);
plot(xi(1:4),yi(1:4),'or'); hold on,
plot(xT,yT,'+r'); hold on,
contour(X,Y,GMAP4,'ShowText','on');
legend('landmark location of the object',' true location of the object','MAP objective function contours'), 
title('the MAP objective function contours for K = 4'),
xlabel('x'), ylabel('y')

