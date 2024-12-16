
% %% ��������������
% % �ٸ����ӣ�dataΪm*1���������̼�.
% clc;
% clear;
% data=xlsread('data.xlsx');
% % % data=data(:,2);
% N=length(data);%�ܹ��۲�ֵ
% M=240;%����Ƶ�ʣ���1min��Mȡ240,2min��ȡ120......���й�һ���Ƶ�Ĺ۲�ֵΪ240.
% S=reshape(data,M,N/M);
% S=[S(1,:);S];
% S=S';

%% �ȼ���Ƶ�ʣ��ڼ���RV��RBV������Ϊ3186*241,3186Ϊ�۲�ֵ��241Ϊ1��ĸ�Ƶ���ݸ��������������ݡ� 
NN = length(S);
l=391;%1��Ĺ۲�ֵ%����391
 M=78;%�Ѿ�����������ų���Ƶ��,��5min ��ΪM=48
for day = 1:NN 
       tempt = S(day,1:5:l);    %%%��ʱ�洢ÿ��İ�K(i)���ӳ����ļ۸�����
        rt = price2ret(tempt);	%%%��ʱ�洢ÿ��İ�1���ӳ����Ķ�������������
        RET(day)=sum(rt);
     %%%������ʵ�ֲ�����
        RV(day) = sum(rt.^2);%��ʵ�ֲ�����
        RQ(day) = (M/3)*sum(rt.^4);%��ʵ�ֲ�����
        RV_z(day)=sum(rt(find(rt>0)).^2);
        RV_f(day)=sum(rt(find(rt<0)).^2);

     %%%������ʵ��˫�ݴα�����
      for j=1:(M-1)
        rr(j) = abs(rt(j))*abs(rt(j+1));
      end
      RBV (day)= ((0.79788)^(-2))*sum(rr);%��ʵ��˫�ݴα��
     %%%������ʵ�����ݴα�����
       for j=1:(M-3)
        rrr(j) = (abs(rt(j))*abs(rt(j+1))*abs(rt(j+2))).^(4/3);
      end
      TP(day)= (2^(2/3)*(gamma(7/6)/gamma(0.5)))^(-3)*M*(M/(M-2))*sum(rrr);%��ʵ��˫�ݴα��
      
end
 RV_z= RV_z';
  RV_f= RV_f';
%% ���⣬������OX��������ĳ����Ƿ���ȷ����ȻҲ������OX����RV��BPV��TP
% %% ���뵽oxmetrics ����TP�������ݱ�������
% 
% for day=1:3186
%     Ox(day,:) = S(day,1:2:l);%2��ʾ����Ƶ��
% end
% %% ����������(���뵽Ox)
%  rrr = price2ret(Ox');
%  rrr=rrr';
% xlswrite('oxmetrics',rrr);

% %% ��ox����õ�TP
% .........���뵽Matlab���棬����ΪTP
% % RV������ʵ�ֲ�����
% % RBV��������ݲ�
% %TP Ϊ�����ݲ�
%     
%% ����Zͳ��ֵ������Huang and Tauchen (2005��

% Z=((RV-RBV)/RV)/((0.5*pi)^2+pi-5)*1/M*max(1,TP/RBV^2)
RV=RV';
RBV=RBV';
TP=TP';
Z1=(RV-RBV)./RV;
Z2=[(0.5*pi)^2+pi-5]/M;
Z3=TP./(RBV.^2)
Z4=ones(NN,1,1);
Z5=max(Z4,Z3);
Z=Z1./(Z2*Z5).^0.5;%ΪZͳ��ֵ


%% ���㹱��ֵ
%%% ָʾ�ͺ���
ZS=Z-3.090232306;%1.65Ϊ��׼��̬�ֲ���0.95��Ӧ��ͳ��ֵ;0.99��Ӧ2.326347874��0999��3.090232306
 

g = find(ZS>0);%���Ǽ�¼x�б���������
ZS(g) = 1;%�ҳ������������Ҹ�1����������
t = find(ZS<=0);%���Ǽ�¼x�б���������
ZS(t) = 0;%�ҳ������������Ҹ�1����������

zeros=zeros(NN,1);
ZMAR=max(RV-RBV,0);
Z_J=ZS.*ZMAR;
Z_RJ=Z_J./RV;


%% HAR-RV-J model 

RV;%��ʵ�ֲ�����
n=length(RV);
RV_W=zeros(n,1);
RV_M=zeros(n,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%���㲨��������
% RV(t+1),RV(t+5),RV(t+22)
RVt=RV(2:end,1);
% RVt5=RV(6:end,1);
% RVt22=RV(23:end,1);

% ����RV_W
for i=1:(n-4)
    RV_W(4+i,1)=mean(RV(i:i+4,1));
end
RV_W;

% ����RV_M
for i=1:(n-21)
    RV_M(21+i,1)=mean(RV(i:i+21,1));
end
RV_M;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%������Ծ����  J=max(RV-RBV,0)
J=max(RV-RBV,zeros);
[i,v]=find(J>0);%���Կ������ٸ�����0�����֡�

%% ����ģ��HAR-RV-CJ,Andersen.et al 2007


CJ=ZS.*(RV-RBV);%��Ծ
ZSS=1-ZS;
CRV=ZS.*RBV+ZSS.*RV;
% CRV_z=ZS.*RBV+ZSS.* RV_z;%semi-v
% CRV_f=ZS.*RBV+ZSS.* RV_f;%semi-v
CRVt=CRV(2:end,1);
CRVt5=CRV(6:end,1);
CRVt22=CRV(23:end,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%���㲨�������� 
% ����CRV_W
for i=1:(n-4)
    CRV_W(4+i,1)=mean(CRV(i:i+4,1));
end
CRV_W;

% ����CRV_M
for i=1:(n-21)
    CRV_M(21+i,1)=mean(CRV(i:i+21,1));
end
CRV_M;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ������Ծ����
% ����CJ_W
for i=1:(n-4)
    CJ_W(4+i,1)=mean(CJ(i:i+4,1));
end
CJ_W;

% ����CJ_M
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
% % ����CRV_M
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
% % ����CRV_M
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
