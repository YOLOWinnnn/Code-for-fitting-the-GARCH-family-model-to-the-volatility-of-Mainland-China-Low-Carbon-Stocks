%data=xlsread('C:\Users\Chen\Desktop\Ԥ��ֵ.xlsx');
data=data;
ture=data(:,1);
T=size(data,1);
MSPE=zeros(size(data,2)-1,1);%��һ��Ϊbench���ڶ��к�Ϊmodel��MSPE
RRRR=zeros(size(data,2)-2,1);
for i=2:size(data,2)
    MSPE(i-1)=sum((ture-data(:,i)).^2)/T;
end
for i=2:length(MSPE)
   RRRR(i-1)=1-MSPE(i)/MSPE(1);
end



