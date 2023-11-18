function p = Predict(Theta1, Theta2, X)

p = zeros(size(X, 2), 1);   % Predict result
B = size(X, 2);     % test sample size
X = [ones(1, B); X];      % test sample

a1 = X;
z2 = Theta1 * a1;
a2 = ReLU(z2);       % hidden layer node
a2 = [ones(1, B); a2];
z3 = Theta2 * a2;
h = ReLU(z3);       %output layer node

[~, p] = max(h, [], 1);
p = p - 1;      % Predict result

end
