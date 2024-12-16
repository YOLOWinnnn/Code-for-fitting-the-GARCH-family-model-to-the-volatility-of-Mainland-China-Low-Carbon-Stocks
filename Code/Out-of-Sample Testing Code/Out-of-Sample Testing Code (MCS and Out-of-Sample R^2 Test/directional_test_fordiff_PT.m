% the directional accuracy of the forecasts
%  based on Pesaran Timmermann (2009) Test
% suitable for the forecasts of returns, difference of logs,percarntage change,...
% Tests of the null of no directional accuracy are conducted using the test of Pesaran and Timmermann (2009).
% Written by Yaojie Zhang, any error, please contact at yaojie.zhang@outlook.com
% First version: 2017/11, Last modified: 2017/11   --**--tested--**--
function [success_ratio, Svalue, p] = directional_test_fordiff_PT(actual,forecast)

% Input:
% actual   = T-vector of actual y
% forecast = T by n matric of the forecasts of y, n is the No. of forecasting methods
% Output:
% success_ratio Svalue p, obviously...


%%% check input
T=size(actual,1);
[T1,n]=size(forecast);
if T~=T1
    error('"actual" and "forecast" must have same # obs.')
end
%%% begin to calculate output
direction_sign=zeros(T,n);  % 0 means wrong, 1 means right.
product_of_actual_forecast=nan(T,n);
for i=1:n
    product_of_actual_forecast(:,i)=actual.*forecast(:,i);
    direction_sign(product_of_actual_forecast>0)=1;
    %product_of_actual_forecast>0, which mean both upward or both downward
    
end
success_ratio=mean(direction_sign)';  %  Namely, P
Py=mean(actual>0);
Px=mean(forecast>0);
Pstar=Py.*Px+(1-Py).*(1-Px);
for i=1:n
    VP=Pstar(i)*(1-Pstar(i))/T;
    VPstar=(((2*Py-1)^2*Px(i)*(1-Px(i)))/T)+(((2*Px(i)-1)^2*Py*(1-Py))/T)...
        +((4*Px(i)*Py*(1-Px(i))*(1-Py))/(T^2));
    Svalue(i,1)=(success_ratio(i)-Pstar(i))/sqrt(VP-VPstar);
    p(i,1)=1-normcdf(Svalue(i,1));
end
    