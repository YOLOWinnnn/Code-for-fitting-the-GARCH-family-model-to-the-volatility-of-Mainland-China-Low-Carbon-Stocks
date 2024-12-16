%%% Written by Yaojie Zhang; If you find any problem in this code, please send E-mail to yaojie.zhang@outlook.com. Thanks!
%%% First version: 2017/05; Last revised data:2017/7/19
function [f_MEAN,f_MEDIAN,f_TMC,f_DMSPE1,f_DMSPE2,f_BEST1,f_BEST2,f_TMC_last, ...
    w_DMSPE1_out, w_DMSPE2_out,w_TMC_out, w_TMC_last_out]=...
    comb_forecast(f_model,f_actual,theta1,theta2)
%%%自己定义的组合预测的函数
%%%  output：
%      f_MEAN:   mean
%      f_MEDIAN: median
%      f_TMC:    the trimmed mean combination forecast  the largest and smallest trimmed
%      f_DMSPE1: forecast of Discounted mean squared prediction error (DMSPE) with theta1
%      f_DMSPE2: forecast of Discounted mean squared prediction error (DMSPE) with theta2
%      f_BEST1:  The best forecast with theta1
%      f_BEST2:  The best forecast with theta2
%      f_TMC_last:  the trimmed mean combination forecast the worst trimmed only considering the last forecast
%      w_DMSPE_out, w_TMC1_out,w_TMC2_out 权重 (i,j)第i期第j个模型的权重

%%%  input：
%     f_model：若干个模型的预测值，f_model(i,j)：第i期，第j个模型的预测值
%     f_actual：真实值,与 f_model齐次
%     theta：参数，一般为0.9或1，有点像decay factor

%%% Check input
if nargin<=2
    theta1=1;
    theta2=0.9;
end
if nargin<=3
    theta2=0.9;
end
if max(max(isnan(f_model)))==1  %存在NaN
%     error('Input cantains NaN!');
    prompt={'Please input a number to repalce NaN in forecasts. The number is'};
    defans={'0'};
    replaced_number1=inputdlg(prompt,'Forecasts contain NaN. Please repalce it.',1,defans);
    replaced_number2=replaced_number1{1};
    f_model(isnan(f_model))=replaced_number2;
end

[n,m]=size(f_model); %%n期，m个模型
n_actual=size(f_actual,1);
if n~=n_actual
     error('The true forecasts and other forecasts must have same # obs.');
end

error_square=(f_model-repmat(f_actual,1,m)).^2;%%error_square(i,j)：第j个模型，第i期的误差方

w_DMSPE1_out=[];
w_DMSPE2_out=[];
w_TMC_out=[];
w_TMC_last_out = [];

%%%------------The mean combination forecast
f_MEAN=mean(f_model,2);

%%%%-----------the median combination forecast
f_MEDIAN=median(f_model,2);

%%%-----------the trimmed mean combination forecast
for i=1:n
    w_TMC=1./(m-2)*ones(m,1);
    index_max=find(f_model(i,:)==max(f_model(i,:)));
    index_min=find(f_model(i,:)==min(f_model(i,:)));
    if length(index_max)>=2  %%如果有两个以上最大或最小值。只删一个
        index_max=index_max(1);
    end
    if length(index_min)>=2
        index_min=index_min(1);
    end
    w_TMC(index_max,1)=0;
    w_TMC(index_min,1)=0;
    f_TMC(i,1)=f_model(i,:)*w_TMC;
    w_TMC_out=[w_TMC_out;w_TMC'];
end



%%%----------Discounted mean squared prediction error (DMSPE)-----
%%%----DMSPE1--BEST1--%%%  ---DMSPE2--BEST2--

for i=1:n
    if i==1
        w_DMSPE1=1./m*ones(m,1);  %%%第一期没有历史表现根据，按均值计算
        w_DMSPE2=1./m*ones(m,1);  %%%第一期没有历史表现根据，按均值计算
        phi1(1:m,1)=0;
        phi2(1:m,1)=0;
    else
        for j=1:m
            phi1(j,1)=error_square(i-1,j)+theta1*phi1(j,1);
            phi2(j,1)=error_square(i-1,j)+theta2*phi2(j,1);
            %%%%%------*****------上下两种计算方式是一致的（个人觉得上面的好，下面的有重复计算）------*****------
            %             phi1(j,1)=sum(theta1.^(sort(randperm(i-1),'descend'))'.*error_square(1:i-1,j)./theta1);
            %             %%最后除以theta是因为都多乘以了一个theta，因为最近的是不用乘theta的
            %             %%其实最后不除以这个theta，对求最后的权重w没有影响
            %             phi2(j,1)=sum(theta2.^(sort(randperm(i-1),'descend'))'.*error_square(1:i-1,j)./theta2);
            
        end
        
        sum_phi1=sum(1./phi1); %%注意：1当error_square(1:i-1,j)为0，即前i-1期都完美预测时，1/0=NaN,
        sum_phi2=sum(1./phi2);  %%%   2当error_square(1:i-1,j)之和，即sum_phi为0时，下面要除以它，也会出现NaN
        %%%  %               当然这两种情况几乎不可能发生
        
        for j=1:m
            w_DMSPE1(j,1)=1./phi1(j,1)/sum_phi1;
            w_DMSPE2(j,1)=1./phi2(j,1)/sum_phi2;
        end
    end
    f_DMSPE1(i,1)=f_model(i,:)*w_DMSPE1;
    w_DMSPE1_out=[w_DMSPE1_out;w_DMSPE1'];
    index_best1=find(w_DMSPE1==max(w_DMSPE1));
    if length(index_best1)>=2  %%%当最好的有两个以上时，随便取第一个最好的
        index_best1=index_best1(1);%%%比如当i=1时，w_DMSPE都相等，取第一个模型
    end
    f_BEST1(i,1)=f_model(i,index_best1);
    
    f_DMSPE2(i,1)=f_model(i,:)*w_DMSPE2;
    w_DMSPE2_out=[w_DMSPE2_out;w_DMSPE2'];
    index_best2=find(w_DMSPE2==max(w_DMSPE2));
    if length(index_best2)>=2
        index_best2=index_best2(1);
    end
    f_BEST2(i,1)=f_model(i,index_best2);
end


%%%%-------好像不是常规的，但有人用这个the trimmed mean combination forecast
for i=1:n
    if i==1
        w_TMC_last=1./m*ones(m,1);  %%%第一期没有历史表现根据，按均值计算
    else
        w_TMC_last=1./(m-1)*ones(m,1);
        index_worst=find(error_square(i-1,:)==max(error_square(i-1,:)));
        
        w_TMC_last(index_worst,1)=0;
    end
    f_TMC_last(i,1)=f_model(i,:)*w_TMC_last;
    w_TMC_last_out=[w_TMC_last_out;w_TMC_last'];
end
