%ALL_data=[data Jumps Ret];%data:RVt RV RVW RVM;Jumps:J CJ CJW CJM;RET:R1
%R5 R22;
data=log(data);
RVt=data(:,1);RV=data(:,2);RVW=data(:,3);RVM=data(:,4);X=data(:,5);o=data(:,6);
ow=data(:,7);om=data(:,8);
N=length(RV);
k=1000;%% Ԥ��ֵ�ĸ���											
h=1;%hΪ��ʾone-ahead-step  
l=1494; %����Ԥ�ⴰ�ڣ���1:1600 Ԥ��1601����Ϊ1600-1=1599�����l=1599
FM1=zeros(k,1);FM2=zeros(k,1);FM3=zeros(k,1);FM4=zeros(k,1);FM5=zeros(k,1);FM6=zeros(k,1);FM7=zeros(k,1);FM8=zeros(k,1);
for i = 1:k %% 1ΪԤ�����
    dep = RV((i+h):(l+i)); %1��ʾone-ahead-step    
     Y = dep;     %%%ȡK+1���۲�ֵ����ǰK������ģ�ͣ����һ������Ԥ��Ϳ�Ԥ���Ч����
    X0 = ones(length(Y),1);
    
            indep = [ ones(size(Y, 1), 1), RV(i:(l+i-h)) RVW(i:(l+i-h)) RVM(i:(l+i-h)) ];
            b = regress(dep, indep);
             FM1(i)= b(1)+b(2)*RV(l+i)+b(3)*RVW(l+i)+b(4) *RVM(l+i);%Ԥ�� �����һ����ȥ����
            
end
FM1;
for i = 1:k %% 1ΪԤ�����
    dep = RV((i+h):(l+i)); %1��ʾone-ahead-step    
     Y = dep;     %%%ȡK+1���۲�ֵ����ǰK������ģ�ͣ����һ������Ԥ��Ϳ�Ԥ���Ч����
    X0 = ones(length(Y),1);
    
            indep = [ ones(size(Y, 1), 1), RV(i:(l+i-h)) RVW(i:(l+i-h)) RVM(i:(l+i-h)) J(i:(l+i-h))];
            b = regress(dep, indep);
             FM2(i)= b(1)+b(2)*RV(l+i)+b(3)*RVW(l+i)+b(4) *RVM(l+i)+b(5) *J(l+i);%Ԥ�� �����һ����ȥ����
            
end
FM2;
for i = 1:k %% 1ΪԤ�����
    dep = RV((i+h):(l+i)); %1��ʾone-ahead-step    
     Y = dep;     %%%ȡK+1���۲�ֵ����ǰK������ģ�ͣ����һ������Ԥ��Ϳ�Ԥ���Ч����
    X0 = ones(length(Y),1);
    
            indep = [ ones(size(Y, 1), 1), RV(i:(l+i-h)) RVW(i:(l+i-h)) RVM(i:(l+i-h)) CJ(i:(l+i-h)) CJW(i:(l+i-h)) CJM(i:(l+i-h))];
            b = regress(dep, indep);
             FM3(i)= b(1)+b(2)*RV(l+i)+b(3)*RVW(l+i)+b(4) *RVM(l+i)+b(5) *CJ(l+i)+b(6) *CJW(l+i)+b(7) *CJM(l+i);%Ԥ�� �����һ����ȥ����
            
end
FM3;
for i = 1:k %% 1ΪԤ�����
    dep = RV((i+h):(l+i)); %1��ʾone-ahead-step    
     Y = dep;     %%%ȡK+1���۲�ֵ����ǰK������ģ�ͣ����һ������Ԥ��Ϳ�Ԥ���Ч����
    X0 = ones(length(Y),1);
    
            indep = [ ones(size(Y, 1), 1), RV(i:(l+i-h)) RVW(i:(l+i-h)) RVM(i:(l+i-h)) CJ(i:(l+i-h)) CJW(i:(l+i-h)) CJM(i:(l+i-h)) r1(i:(l+i-h)) r5(i:(l+i-h)) r22(i:(l+i-h))];
            b = regress(dep, indep);
             FM4(i)= b(1)+b(2)*RV(l+i)+b(3)*RVW(l+i)+b(4) *RVM(l+i)+b(5) *CJ(l+i)+b(6) *CJW(l+i)+b(7) *CJM(l+i)+b(8) *r1(l+i)+b(9) *r5(l+i)+b(10) *r22(l+i);%Ԥ�� �����һ����ȥ����
            
end
FM4;
for i = 1:k %% 1ΪԤ�����
    dep = RV((i+h):(l+i)); %1��ʾone-ahead-step    
     Y = dep;     %%%ȡK+1���۲�ֵ����ǰK������ģ�ͣ����һ������Ԥ��Ϳ�Ԥ���Ч����
    X0 = ones(length(Y),1);
    
            indep = [ ones(size(Y, 1), 1), RV(i:(l+i-h)) RVW(i:(l+i-h)) RVM(i:(l+i-h)) X(i:(l+i-h))];
            b = regress(dep, indep);
             FM5(i)= b(1)+b(2)*RV(l+i)+b(3)*RVW(l+i)+b(4) *RVM(l+i)+b(5) *X(l+i);%Ԥ�� �����һ����ȥ����
            
end
FM5;
for i = 1:k %% 1ΪԤ�����
    dep = RV((i+h):(l+i)); %1��ʾone-ahead-step    
     Y = dep;     %%%ȡK+1���۲�ֵ����ǰK������ģ�ͣ����һ������Ԥ��Ϳ�Ԥ���Ч����
    X0 = ones(length(Y),1);
    
            indep = [ ones(size(Y, 1), 1), RV(i:(l+i-h)) RVW(i:(l+i-h)) RVM(i:(l+i-h)) J(i:(l+i-h)) X(i:(l+i-h))];
            b = regress(dep, indep);
             FM6(i)= b(1)+b(2)*RV(l+i)+b(3)*RVW(l+i)+b(4) *RVM(l+i)+b(5) *J(l+i)+b(6) *X(l+i);%Ԥ�� �����һ����ȥ����
            
end
FM6;
for i = 1:k %% 1ΪԤ�����
    dep = RV((i+h):(l+i)); %1��ʾone-ahead-step    
     Y = dep;     %%%ȡK+1���۲�ֵ����ǰK������ģ�ͣ����һ������Ԥ��Ϳ�Ԥ���Ч����
    X0 = ones(length(Y),1);
    
            indep = [ ones(size(Y, 1), 1), RV(i:(l+i-h)) RVW(i:(l+i-h)) RVM(i:(l+i-h)) CJ(i:(l+i-h)) CJW(i:(l+i-h)) CJM(i:(l+i-h)) X(i:(l+i-h))];
            b = regress(dep, indep);
             FM7(i)= b(1)+b(2)*RV(l+i)+b(3)*RVW(l+i)+b(4) *RVM(l+i)+b(5) *CJ(l+i)+b(6) *CJW(l+i)+b(7) *CJM(l+i)+b(8) *X(l+i);%Ԥ�� �����һ����ȥ����
            
end
FM7;
for i = 1:k %% 1ΪԤ�����
    dep = RV((i+h):(l+i)); %1��ʾone-ahead-step    
     Y = dep;     %%%ȡK+1���۲�ֵ����ǰK������ģ�ͣ����һ������Ԥ��Ϳ�Ԥ���Ч����
    X0 = ones(length(Y),1);
    
            indep = [ ones(size(Y, 1), 1), RV(i:(l+i-h)) RVW(i:(l+i-h)) RVM(i:(l+i-h)) CJ(i:(l+i-h)) CJW(i:(l+i-h)) CJM(i:(l+i-h)) r1(i:(l+i-h)) r5(i:(l+i-h)) r22(i:(l+i-h)) X(i:(l+i-h))];
            b = regress(dep, indep);
             FM8(i)= b(1)+b(2)*RV(l+i)+b(3)*RVW(l+i)+b(4) *RVM(l+i)+b(5) *CJ(l+i)+b(6) *CJW(l+i)+b(7) *CJM(l+i)+b(8) *r1(l+i)+b(9) *r5(l+i)+b(10) *r22(l+i)+b(11) *X(l+i);%Ԥ�� �����һ����ȥ����
            
end
FM8;
FM=[FM1 FM2 FM3 FM4 FM5 FM6 FM7 FM8];
forecast=exp(FM);

T=size(forecast,1);
for i=1:size(forecast,2)
    MSPE(i,1)=sum((ture-forecast(:,i)).^2)/T;  
end
for i=2:length(MSPE)
   Roos(i-1,1)=1-MSPE(i)/MSPE(1); % �����б�ʾÿ���µ�Ԥ��ģ��(M1~M4)�ͻ�׼ģ��M0�Ƚϵ�������R��ֵ���б�ʾ���ʹ��и��� 
end

forecast2=forecast(:,2:end);  % forecast2����ֻ������ģ��M1~M4
for i=1:size(forecast2,2)
[MSPE_adjusted,p_value]=Perform_CW_test(ture,forecast(:,1),forecast2(:,i));
p_value_all(i,1)=p_value;
MSPE_adjusted_all(i,1)=MSPE_adjusted;
end
m=[Roos*100, MSPE_adjusted_all, p_value_all];

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
mean_LOSS_ALL=[mean_LOSS_MSE  mean_LOSS_MAE mean_LOSS_HMSE mean_LOSS_HMAE];
% MCS ����
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
% ���е�Pֵ����
p_mcs_ALL=[p_mcs_QLIKE p_mcs_QLIKE_SQ p_mcs_MSE  p_mcs_MSE_SQ p_mcs_MAE  p_mcs_MAE_SQ p_mcs_HMSE  p_mcs_HMSE_SQ  p_mcs_HMAE ...
     p_mcs_HMAE_SQ];