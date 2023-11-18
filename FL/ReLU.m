function g = ReLU(z)
g = z;
g(find(z<0)) = 0;
end

