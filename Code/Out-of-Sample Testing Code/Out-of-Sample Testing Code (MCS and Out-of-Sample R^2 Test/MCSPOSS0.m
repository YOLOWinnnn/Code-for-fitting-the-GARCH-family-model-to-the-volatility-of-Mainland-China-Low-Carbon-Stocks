
%%RV_ALL��һ����ʵֵ���ڶ���֮��ΪԤ��ֵ

 clc;clear;
 %RV_ALL = xlsread("XNY_h=22.xlsx");
  RV_ALL =xlsread("DT50.xlsx");
%%  Roos������R**2
ture=RV_ALL(:,1); %%��ʵ������ %�ڶ��л�׼ģ��  %%�����к����µ�ģ��
%ture(find(ture==0)) = 1e-08;
forecast=RV_ALL(:,2:end);  % forecast���������׼ģ��MO����ģ��M1~M4

T=size(forecast,1);
for i=1:size(forecast,2)
    MSPE(i,1)=sum((ture-forecast(:,i)).^2)/T; 
end
for i=2:length(MSPE)
   Roos(i-1,1)=1-MSPE(i)/MSPE(1); % �����б�ʾÿ���µ�Ԥ��ģ��(M1~M4)�ͻ�׼ģ��M0�Ƚϵ�������R��ֵ���б�ʾ���ʹ��и��� 
end

 %%  P_value
 forecast2=forecast(:,2:end);  % forecast2����ֻ������ģ��M1~M4
 for i=1:size(forecast2,2)
 [MSPE_adjusted,p_value]=Perform_CW_test(ture,forecast(:,1),forecast2(:,i));
 p_value_all(i,1)=p_value;
 MSPE_adjusted_all(i,1)=MSPE_adjusted;
 end
 %m=[Roos*100, MSPE_adjusted_all, p_value_all];
 m=[Roos*100, MSPE_adjusted_all, p_value_all];

%% MCS
actual=ture; 
FC=forecast;
N_FC=size(FC,2);
LOSS_QLIKE=log(FC)+repmat( actual ,1 , N_FC )./FC;%QLIKE
mean_LOSS_QLIKE=mean(LOSS_QLIKE)';
LOSS_MSE=(FC-repmat( actual ,1 , N_FC )).^2;%MSE
mean_LOSS_MSE=mean(LOSS_MSE)';
LOSS_MAE=abs(FC-repmat( actual ,1 , N_FC ));%MAE
mean_LOSS_MAE=mean(LOSS_MAE)';
LOSS_HMSE=((1-FC./repmat( actual ,1 ,N_FC ))).^2;%HMSE
mean_LOSS_HMSE=mean(LOSS_HMSE)';
LOSS_HMAE=abs((1-FC./repmat( actual ,1 , N_FC )));  %HMAE
mean_LOSS_HMAE=mean(LOSS_HMAE)';
% ���е���ʧ��������
mean_LOSS_ALL=[mean_LOSS_QLIKE mean_LOSS_MSE  mean_LOSS_MAE mean_LOSS_HMSE mean_LOSS_HMAE];
% MCS ����,mcs(LOSS_QLIKE,0.0001,10000,25)
[includedR1,pvalsR1,excludedR1,includedSQ1,pvalsSQ1,excludedSQ1]=mcs(LOSS_QLIKE,0.0001,10000,2);%QLIKE
[p_mcs_QLIKE] = mcs_resort(includedR1,pvalsR1,excludedR1);
[p_mcs_QLIKE_SQ] = mcs_resort(includedSQ1,pvalsSQ1,excludedSQ1);
[includedR1,pvalsR1,excludedR1,includedSQ1,pvalsSQ1,excludedSQ1]=mcs(LOSS_MSE,0.0001,10000,2);%MSE
[p_mcs_MSE] = mcs_resort(includedR1,pvalsR1,excludedR1);
[p_mcs_MSE_SQ] = mcs_resort(includedSQ1,pvalsSQ1,excludedSQ1);
[includedR1,pvalsR1,excludedR1,includedSQ1,pvalsSQ1,excludedSQ1]=mcs(LOSS_MAE,0.0001,10000,2);%MAE
[p_mcs_MAE] = mcs_resort(includedR1,pvalsR1,excludedR1);
[p_mcs_MAE_SQ] = mcs_resort(includedSQ1,pvalsSQ1,excludedSQ1);
[includedR1,pvalsR1,excludedR1,includedSQ1,pvalsSQ1,excludedSQ1]=mcs(LOSS_HMSE,0.0001,10000,2);%HMSE
[p_mcs_HMSE] = mcs_resort(includedR1,pvalsR1,excludedR1);
[p_mcs_HMSE_SQ] = mcs_resort(includedSQ1,pvalsSQ1,excludedSQ1);
[includedR1,pvalsR1,excludedR1,includedSQ1,pvalsSQ1,excludedSQ1]=mcs(LOSS_HMAE,0.0001,10000,2); %HMAE
[p_mcs_HMAE] = mcs_resort(includedR1,pvalsR1,excludedR1);
[p_mcs_HMAE_SQ] = mcs_resort(includedSQ1,pvalsSQ1,excludedSQ1);
% ���е�Pֵ���� zeros(20,1)*nan
%p_mcs_ALL = [p_mcs_QLIKE p_mcs_MSE p_mcs_MAE p_mcs_HMSE  p_mcs_HMAE  p_mcs_QLIKE_SQ  p_mcs_MSE_SQ p_mcs_MAE_SQ p_mcs_HMSE_SQ p_mcs_HMAE_SQ];
% ���е�Pֵ����
p_mcs_ALL=[p_mcs_QLIKE p_mcs_QLIKE_SQ p_mcs_MSE  p_mcs_MSE_SQ p_mcs_MAE  p_mcs_MAE_SQ p_mcs_HMSE  p_mcs_HMSE_SQ  p_mcs_HMAE ...
     p_mcs_HMAE_SQ];
%%   Direction-of-Change or success ratio
actual_DOC=diff(actual);
FC_DOC=FC(2:end,1:end)-repmat(actual(1:end-1,1),1,N_FC);
[success_ratio, Svalue, p_SR] = directional_test_fordiff_PT(actual_DOC,FC_DOC);
success_ratio_ALL=[success_ratio, Svalue, p_SR];

%%  DM test
e11=LOSS_MSE(:,1); 
e12=LOSS_MAE(:,1);
e13=LOSS_HMSE(:,1);
e14=LOSS_HMAE(:,1);
e15=LOSS_QLIKE(:,1);

for i=2:size(LOSS_MSE,2)
e2=LOSS_MSE(:,i);
e3=LOSS_MAE(:,i);
e4=LOSS_HMSE(:,i);
e5=LOSS_HMAE(:,i);
e6=LOSS_QLIKE(:,i);

[DM1 p_DM1] = dmtest(e11, e2, 1);
[DM2 p_DM2] = dmtest(e12, e3, 1);
[DM3 p_DM3] = dmtest(e13, e4, 1);
[DM4 p_DM4] = dmtest(e14, e5, 1);
[DM5 p_DM5] = dmtest(e15, e6, 1);
DM_all1(i-1,1)=DM1;    p_DM_all1(i-1,1)=p_DM1;
DM_all2(i-1,1)=DM2;    p_DM_all2(i-1,1)=p_DM2; 
DM_all3(i-1,1)=DM3;    p_DM_all3(i-1,1)=p_DM3; 
DM_all4(i-1,1)=DM4;    p_DM_all4(i-1,1)=p_DM4; 
DM_all5(i-1,1)=DM5;    p_DM_all5(i-1,1)=p_DM5;
end
DM_all=[DM_all5 DM_all1 DM_all2 DM_all3 DM_all4];
p_DM_all=[p_DM_all5 p_DM_all1 p_DM_all2 p_DM_all3 p_DM_all4];


%ave_utility(:,:) = 100*Asset_allocatation_ConstantSR(actual,FC,0.08);
%utility = 100*Asset_allocatation_ConstantSR(actual,FC,0.08);

