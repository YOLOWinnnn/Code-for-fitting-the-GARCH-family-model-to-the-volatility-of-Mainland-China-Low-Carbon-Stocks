% Use PLS (Partial least-squares) recursively to construct predictors,which is used for
% out-of-sample forecast. Note: if not, recursively, it used for in-sample
% test, and the in-sample test contains look-ahead bias. 
% like the way of pca
% Author: Yaojie Zhang, any error, please contact at yaojie.zhang@outlook.com
% First version: 2017/11, Last modified: 2017/11
function [x_pls] = pls2predictor_out(y,x)
% Input:
% y   = dependent variable, T-vector
% x   = T by N matric of the raw predictors

% Output:
% x_pls  = PLS predictors  (look-ahead bias-free predictor)

% Note： This version is only suitable for one-step ahead forecast 预测一次用一次
% Reference:
% Kelly, B., Pruitt, S., 2013. Market expectations in the cross‐section of present values.
% The Journal of Finance 68, 1721-1756.
% Kelly, B., Pruitt, S., 2015. The three-pass regression filter:
% A new approach to forecasting using many predictors. Journal of Econometrics 186, 294-316.
% Huang, D., Jiang, F., Tu, J., Zhou, G., 2014. Investor sentiment aligned:
% A powerful predictor of stock returns. Review of Financial Studies 28, 791-837.

%%% check input
T=size(y,1);
[T1,N]=size(x);
if T~=T1
    error('Y and X must have same # obs.')
end


for i=1:N
    B=regress( x(1:end-1,i),[ones(T-1,1) y(2:end)]);
    beta(i,1)=B(2);
end
for t=1:T
    B2 =regress( x(t,1:N)',[ones(N,1) beta]);
    x_pls(t,1)=B2(2);
end

