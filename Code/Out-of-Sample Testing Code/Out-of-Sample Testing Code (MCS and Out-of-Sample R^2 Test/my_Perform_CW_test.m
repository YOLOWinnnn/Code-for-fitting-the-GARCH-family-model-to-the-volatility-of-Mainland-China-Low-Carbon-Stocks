function [MSPE_adjusted,p_value]=my_Perform_CW_test(data)
%data 的第一列为真实值，第二列为bench，第三列到最后都是预测值
%输出的参数为第三列到最后一列的检验统计量p_value


actual=data(:,1);
bench=data(:,2);
MSPE_adjusted=zeros(size(data,2)-2,1);
p_value=MSPE_adjusted;
for i=3:size(data,2)
    forecast_2=data(:,i);
    e_1=actual-bench;
    e_2=actual-forecast_2;
    f_hat=e_1.^2-(e_2.^2-(bench-forecast_2).^2);
    Y_f=f_hat;
    X_f=ones(size(f_hat,1),1);
    beta_f=(inv(X_f'*X_f))*(X_f'*Y_f);
    e_f=Y_f-X_f*beta_f;
    sig2_e=(e_f'*e_f)/(size(Y_f,1)-1);
    cov_beta_f=sig2_e/(X_f'*X_f);
    MSPE_adjusted(i-2)=beta_f/sqrt(cov_beta_f);
    p_value(i-2)=1-normcdf(MSPE_adjusted(i-2),0,1);                               
end