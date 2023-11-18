function g = Softmax(z)

ExpSum = sum(exp(z),1);
tmp=repmat(ExpSum,size(z,1),1);
g = exp(z) ./ tmp;

end

