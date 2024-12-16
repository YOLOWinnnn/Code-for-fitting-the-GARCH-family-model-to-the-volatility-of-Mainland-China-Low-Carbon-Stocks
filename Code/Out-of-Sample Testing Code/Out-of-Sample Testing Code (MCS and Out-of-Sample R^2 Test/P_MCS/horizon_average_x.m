%%% horizon_average; used to construct independent variable, x
%%% Written by Yaojie Zhang  (张耀杰); If you find any problem in this code, please send E-mail to yaojie.zhang@outlook.com. Thanks!
%%% First version: 2017/07/31; Last revised data:2017/7/31
function data_h=horizon_average_x(data,h)
%%% take average of data in the horizon of h.
%%data是原始的日行情数据,h想要的周期，例如h=5,想要取近五天的均值,
%%%   第t行的取均值变量，only use the data up to t. 前h-1行为nan

[n,m]=size(data); %n天，m个资产序列

data_h=nan(n,m); %先构造，节约算法时间

for j=1:m
    for i=h:n
        data_h(i,j)=mean(data(i-h+1:i,j));
    end    
end




