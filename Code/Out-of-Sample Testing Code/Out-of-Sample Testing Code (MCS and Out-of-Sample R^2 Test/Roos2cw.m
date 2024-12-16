%data=xlsread('C:\Users\Chen\Desktop\预测值.xlsx');
% ture=data(:,1);
% T=size(data,1);
% MSPE=zeros(size(data,2)-1,1);%第一行为bench，第二行后为model的MSPE
% RRR2=zeros(size(data,2)-2,1);
% cw=RRR2;
% for i=2:size(data,2)
%     MSPE(i-1)=sum((ture-data(:,i)).^2)/T;
% end
% for i=2:length(MSPE)
%     RRR2(i-1)=1-MSPE(i)/MSPE(1);
%     cw(i-1)=MSPE(1)-MSPE(i)+sum((data(:,2)-data(:,i+1)).^2)/T;
% end

ture=data(:,1);
T=size(data,1);
MSPE=zeros(size(data,2)-1,1);%第一行为bench，第二行后为model的MSPE
RRR2=zeros(size(data,2)-2,1);
cw=RRR2;
for i=2:size(data,2)
    MSPE(i-1)=sum((ture-data(:,i)).^2)/T;
end
for i=2:length(MSPE)
    RRR2(i-1)=1-MSPE(i)/MSPE(1);
    cw(i-1)=MSPE(1)-MSPE(i)+sum((data(:,2)-data(:,i+1)).^2)/T;
end
