function g = ReLUGradient(z)

g = zeros(size(z));
g(find(z>=0)) = 1;

end