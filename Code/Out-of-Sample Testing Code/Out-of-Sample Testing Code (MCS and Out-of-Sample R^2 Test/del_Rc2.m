function Rc2=del_Rc2(data,I)
%data 第一列为ture 第二列为bench，从第三列到最后一列为预测值
Rc2=zeros(size(data,2)-2,1);
ture=data(:,1);
bench=data(:,2);
for i=3:size(data,2)
    Rc2(i-2)=1-sum((ture-data(:,i)).^2.*I)/sum((ture-bench).^2.*I);
end




