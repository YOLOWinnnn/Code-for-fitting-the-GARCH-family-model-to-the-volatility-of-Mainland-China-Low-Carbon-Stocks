
% %% 计算日内收益率
% % 举个例子，data为m*1的日内收盘价.
% clc;
% clear;
% data=xlsread('data.xlsx');
% % % data=data(:,2);
% N=length(data);%总共观测值
% M=240;%抽样频率，如1min则M取240,2min钟取120......如中国一般高频的观测值为240.
% S=reshape(data,M,N/M);
% S=[S(1,:);S];
% S=S';

%% 先计算频率，在计算RV和RBV，矩阵为3186*241,3186为观测值，241为1天的高频数据个数，含开盘数据。 
NN = length(S);
l=391;%1天的观测值%美国391
 M=78;%已经计算出的最优抽样频率,如5min 就为M=48
for day = 1:NN 
       tempt = S(day,1:5:l);    %%%临时存储每天的按K(i)分钟抽样的价格序列
        rt = price2ret(tempt);	%%%临时存储每天的按1分钟抽样的对数收益率序列
        RET(day)=sum(rt);
     %%%计算已实现波动率
        RV(day) = sum(rt.^2);%已实现波动率
        RQ(day) = (M/3)*sum(rt.^4);%已实现波动率
        RV_z(day)=sum(rt(find(rt>0)).^2);
        RV_f(day)=sum(rt(find(rt<0)).^2);

     %%%计算已实现双幂次变差波动率
      for j=1:(M-1)
        rr(j) = abs(rt(j))*abs(rt(j+1));
      end
      RBV (day)= ((0.79788)^(-2))*sum(rr);%已实现双幂次变差
     %%%计算已实现三幂次变差波动率
       for j=1:(M-3)
        rrr(j) = (abs(rt(j))*abs(rt(j+1))*abs(rt(j+2))).^(4/3);
      end
      TP(day)= (2^(2/3)*(gamma(7/6)/gamma(0.5)))^(-3)*M*(M/(M-2))*sum(rrr);%已实现双幂次变差
      
end
 RV_z= RV_z';
  RV_f= RV_f';
%% 另外，可以用OX检验上面的程序是否正确，当然也可以用OX计算RV，BPV，TP
% %% 导入到oxmetrics 进行TP（三次幂变差，）计算
% 
% for day=1:3186
%     Ox(day,:) = S(day,1:2:l);%2表示抽样频率
% end
% %% 计算收益率(导入到Ox)
%  rrr = price2ret(Ox');
%  rrr=rrr';
% xlswrite('oxmetrics',rrr);

% %% 在ox计算得到TP
% .........导入到Matlab里面，命令为TP
% % RV代表已实现波动率
% % RBV代表二次幂差
% %TP 为三次幂差
%     
%% 计算Z统计值，依据Huang and Tauchen (2005）

% Z=((RV-RBV)/RV)/((0.5*pi)^2+pi-5)*1/M*max(1,TP/RBV^2)
RV=RV';
RBV=RBV';
TP=TP';
Z1=(RV-RBV)./RV;
Z2=[(0.5*pi)^2+pi-5]/M;
Z3=TP./(RBV.^2)
Z4=ones(NN,1,1);
Z5=max(Z4,Z3);
Z=Z1./(Z2*Z5).^0.5;%为Z统计值


%% 计算贡献值
%%% 指示型函数
ZS=Z-3.090232306;%1.65为标准正态分布，0.95对应的统计值;0.99对应2.326347874，0999，3.090232306
 

g = find(ZS>0);%这是记录x中比零大的索引
ZS(g) = 1;%找出比零大的数并且赋1，下面类似
t = find(ZS<=0);%这是记录x中比零大的索引
ZS(t) = 0;%找出比零大的数并且赋1，下面类似

zeros=zeros(NN,1);
ZMAR=max(RV-RBV,0);
Z_J=ZS.*ZMAR;
Z_RJ=Z_J./RV;


%% HAR-RV-J model 

RV;%已实现波动率
n=length(RV);
RV_W=zeros(n,1);
RV_M=zeros(n,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%计算波动率序列
% RV(t+1),RV(t+5),RV(t+22)
RVt=RV(2:end,1);
% RVt5=RV(6:end,1);
% RVt22=RV(23:end,1);

% 计算RV_W
for i=1:(n-4)
    RV_W(4+i,1)=mean(RV(i:i+4,1));
end
RV_W;

% 计算RV_M
for i=1:(n-21)
    RV_M(21+i,1)=mean(RV(i:i+21,1));
end
RV_M;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%计算跳跃序列  J=max(RV-RBV,0)
J=max(RV-RBV,zeros);
[i,v]=find(J>0);%可以看看多少个大于0的数字。

%% 计算模型HAR-RV-CJ,Andersen.et al 2007


CJ=ZS.*(RV-RBV);%跳跃
ZSS=1-ZS;
CRV=ZS.*RBV+ZSS.*RV;
% CRV_z=ZS.*RBV+ZSS.* RV_z;%semi-v
% CRV_f=ZS.*RBV+ZSS.* RV_f;%semi-v
CRVt=CRV(2:end,1);
CRVt5=CRV(6:end,1);
CRVt22=CRV(23:end,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%计算波动率序列 
% 计算CRV_W
for i=1:(n-4)
    CRV_W(4+i,1)=mean(CRV(i:i+4,1));
end
CRV_W;

% 计算CRV_M
for i=1:(n-21)
    CRV_M(21+i,1)=mean(CRV(i:i+21,1));
end
CRV_M;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 计算跳跃部分
% 计算CJ_W
for i=1:(n-4)
    CJ_W(4+i,1)=mean(CJ(i:i+4,1));
end
CJ_W;

% 计算CJ_M
for i=1:(n-21)
    CJ_M(21+i,1)=mean(CJ(i:i+21,1));
end
CJ_M;


% HARRV=[RVt,RV,RV_W,RV_M]; %HAR-RV
% HARRVJ=[RVt,RV,RV_W,RV_M,J];%HAR-RV-J
% HARRVCJ=[CRVt,CRV,CRV_W,CRV_M,CJ];%HAR-RV-CJ

SJV=RV_z-RV_f;
%SJV=[RV_z,RV_f,JJ];

L1 = (SJV>0);
L2 = (SJV<0);

SJVz=SJV.*L1;
SJVf=SJV.*L2;

RVt=[RVt;0];
CRVt=[CRVt;0];
HARRV=[RVt,RV,RV_W,RV_M]; %HAR-RV
HARRVJ=[RVt,RV,RV_W,RV_M,J];%HAR-RV-J
HARRVCJ=[RVt,CRV,CRV_W,CRV_M,CJ,CJ_W,CJ_M];%HAR-RV-CJ
% 
zzzzz=[RVt,RV,RV_W,RV_M,CRV,CRV_W,CRV_M];
% HARRV=[RVt,RV,RV_W,RV_M]; %HAR-RV
% HARRVJ=[RVt,RV,RV_W,RV_M,J];%HAR-RV-J
% HARRVCJ=[CRVt,CRV,CRV_W,CRV_M,CJ];%HAR-RV-CJ

% 
% %% HAR-S-RV-J
% for i=1:(n-4)
%    RVW_z(4+i,1)=mean(RV_z(i:i+4,1));
% end
% RVW_z;
% 
% % 计算CRV_M
% for i=1:(n-19)
%    RVM_z(19+i,1)=mean(RV_z(i:i+19,1));
% end
%  RVM_z;
% % 
% 
% for i=1:(n-4)
%    RVW_f(4+i,1)=mean(RV_f(i:i+4,1));
% end
% RVW_f;
% 
% % 计算CRV_M
% for i=1:(n-19)
%    RVM_f(19+i,1)=mean(RV_f(i:i+19,1));
% end
%  RVM_f;
%  
%  HARSRVJ=[RVt, RV_z,RVW_z,RVM_z,RV_f,RVW_f,RVM_f,J];
% % 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
