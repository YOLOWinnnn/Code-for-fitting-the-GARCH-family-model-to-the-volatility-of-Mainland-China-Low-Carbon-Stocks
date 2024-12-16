%%% takeaverage; used to construct dependent variable, y
%%% Written by Yaojie Zhang (张耀杰); 
% If you find any problem in this code, please send E-mail to yaojie.zhang@outlook.com. Thanks!
%%% First version: 2017/08/15; Last revised data:2017/12/15
function y_h=takeaverage_horizon_y(y,h)
% input:
% y: a time series that need to take average in a horizon
% h: horizon
% output: y_h
% y_h 与 y  --**-长度相等 但不是齐次的，y_h领先了h-1期 --**--，不管h是多少，----%%---只是后h-1行没数据---%%----
%%比如h=3,y_h(1)是y(1)y(2)y(3)的均值
T=length(y);
n=length(h);
y_h=nan(T,n);  % if h>1, the last h-1 in y_h is nan.
for j=1:n
    for t=1:T-(h(j)-1)
        y_h(t,j)=mean(y(t:t+(h(j)-1)));
    end
end