%% read in a table of delivery times(y),number of cases(x1),
% and distance(x2)
t = readtable('project1part1.xls');
y = t{:,1};
x1 = t{:,2};
x2 = t{:,3};
%%
x = ones(25,1); %column of 25 ones
X = [x, x1, x2]; % columns of 25 ones, 25 x1 values, 25 x2 values
bhat = (X'*X)^(-1)*X'*y; %calculate the values of bhat
b0 = bhat(1); b1 = bhat(2); b2 = bhat(3); %the values of bhat
yhat = b0 + b1*x1 + b2*x2; %
MSE = 0;
for n = 1:25
    MSE = MSE + (y(n)-yhat(n))^2;
end
MSE = sqrt((1/n)*MSE);
