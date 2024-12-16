% resort MCS result
% Author: Yaojie Zhang  (уер╚╫э), any error, please contact at yaojie.zhang@outlook.com
% First version: 2018/01, Last modified: 2018/01
function [p_sort] = mcs_resort(include_no,p,exclude_no)
% Input:
% include_no,p,exclude_no: the output from mcs
% Output:
% p_sort: the sorted p-values

no_mcs=[exclude_no ; include_no];
N=length(no_mcs);

for i=1:N
    p_sort(i,1)=p(find(no_mcs==i));       
end