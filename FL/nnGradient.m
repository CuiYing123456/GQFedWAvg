function [Theta1_grad, Theta2_grad] = nnGradient(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

B = size(X, 2);         % minibatch size
X = [ones(1, B); X];      % training sample
y_vec = zeros(num_labels, B);     % 0/1 output

for i = 1:B
        y_vec(y(i) + 1, i) = 1;
end

delta1 = zeros(hidden_layer_size, input_layer_size + 1);     %Theta1 gradient accumulate
delta2 = zeros(num_labels, hidden_layer_size + 1);     %Theta2 gradient accumulate

for i = 1 : B
    a1 = X(:, i);      % input layer node
    z2 = Theta1 * a1;
    a2 = ReLU(z2);     % hidden layer node
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = Softmax(z3);     % output layer node
    y_tmp = y_vec(:, i);
    g_z2 = ReLUGradient(z2);   %ReLU gradient
    delta2_tmp = Theta2(:, 2:end)' * SoftmaxGradient(a3, y_tmp) .* g_z2;
    delta1 = delta1 + delta2_tmp*a1';
    delta2 = delta2 + SoftmaxGradient(a3, y_tmp) * a2';
end

Theta1_grad = delta1 / B;
Theta2_grad = delta2 / B;

% regularization gradient
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda*Theta1(:, 2:end) / B;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda*Theta2(:, 2:end) / B;

end
