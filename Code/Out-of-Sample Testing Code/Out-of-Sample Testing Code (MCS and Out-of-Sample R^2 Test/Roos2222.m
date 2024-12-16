%data=xlsread('C:\Users\Chen\Desktop\预测值.xlsx');
data=data;
ture=data(:,1);
T=size(data,1);
MSPE=zeros(size(data,2)-1,1);%第一行为bench，第二行后为model的MSPE
RRRR=zeros(size(data,2)-2,1);
for i=2:size(data,2)
    MSPE(i-1)=sum((ture-data(:,i)).^2)/T;
end
for i=2:length(MSPE)
   RRRR(i-1)=1-MSPE(i)/MSPE(1);
end



