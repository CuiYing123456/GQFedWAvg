function [J] = nnCostFunction(Theta1, Theta2, num_labels, X, y, lambda)

B = size(X, 2);     % minibatch size
X = [ones(1, B); X];      % minibatch sample

a1 = X;       % input layer node
a2 = ReLU(Theta1 * a1);       % hidden layer node
a2 = [ones(1, B); a2];
h = Softmax(Theta2 * a2);       %output layer node


y_vec = zeros(num_labels, B);
for i = 1 : B
        y_vec(y(i) + 1, i) = 1;
end

% relularization cost function
J = sum(sum((- y_vec .* log(h) - (1 - y_vec) .* log(1 - h)))) / B;
J_regular = lambda * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))) / (2 * B);
J = J + J_regular;

end
