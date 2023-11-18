%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
fprintf('Loading Data ...\n')

% load training data set
fid = fopen('./LoadData/train-images.idx3-ubyte', 'rb');
A_train_img = fread(fid, 4, 'uint32', 'ieee-be');
img_train_num = A_train_img(2);
row_num = A_train_img(3);
col_num = A_train_img(4);
img_train = fread(fid, [row_num*col_num, img_train_num], 'unsigned char', 'ieee-be');
img_train = (img_train/255);

% load training label set
fid = fopen('./LoadData/train-labels.idx1-ubyte', 'rb');
A_train_label = fread(fid, 2, 'uint32', 'ieee-be');
lab_train_num = A_train_label(2);
lab_train = fread(fid, lab_train_num, 'unsigned char',  'ieee-be');

% load test data set
fid = fopen('./LoadData/t10k-images.idx3-ubyte', 'rb');
A_test_img = fread(fid, 4, 'uint32', 'ieee-be');
img_test_num = A_test_img(2);
img_test = fread(fid, [row_num*col_num, img_test_num], 'unsigned char',  'ieee-be');
img_test = (img_test/255);

% load test label set
fid = fopen('./LoadData/t10k-labels.idx1-ubyte', 'rb');
A_test_label = fread(fid, 2, 'uint32', 'ieee-be');
lab_test_num = A_test_label(2);
lab_test = fread(fid, lab_test_num, 'unsigned char',  'ieee-be');
lab_test = lab_test';

fclose('all');
%% Setup the parameters
% network hyperparameter
input_layer_size  = 784;  % 28 x 28 Input Images of Digits
hidden_layer_size = 128;   % number of hidden units
num_labels = 10;          % 10 labels, from 1 to 10
N = 10;     % N workers

% local dataset
local_batch_size = img_train_num/N;
local_img = zeros(row_num * col_num, local_batch_size, N);
local_lab = zeros(local_batch_size, N);
for i = 1:N
    mini_idx = [local_batch_size * (i - 1) + 1: local_batch_size * i];
    local_img(:, :, i) = img_train(:, mini_idx);
    local_lab(:, i) = lab_train(mini_idx);
end

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
J_epoch_init = nnCostFunction(initial_Theta1, initial_Theta2, num_labels, img_train, lab_train, lambda);

pred = Predict(initial_Theta1, initial_Theta2, img_train);
pred = pred';
Accuracy_epoch_init = mean(double(pred == lab_train)) * 100;
save(['initial_Theta1.mat'], 'initial_Theta1');
save(['initial_Theta2.mat'], 'initial_Theta2');
save(['J_epoch_init.mat'], 'J_epoch_init');
save(['Accuracy_epoch_init.mat'], 'Accuracy_epoch_init');

% simulation parameters
lambda = 0;     % regularization
lr = 0.01;
B = 1;
Kn = 4*ones(N, 1);
K0 = 1000;
s0t = 2^32-1;
s0 = 2^32-1;
W = 1/N*ones(N, 1);

I = 5;
range = 1;

acc=zeros(K0,I);
J_step=zeros(K0,I);

snt = 2.^(8*ones(N, 1))-1;
sn = 2.^(8*ones(N, 1))-1;
for iter = 1 : I
    fprintf('\nTraining Neural Network... \n')
    Theta1_global=initial_Theta1;
    Theta2_global=initial_Theta2;
    Theta1_local = zeros(size(Theta1_global));
    Theta2_local = zeros(size(Theta2_global));
    QuantizedGradient1 = zeros([size(Theta1_global), N]);
    QuantizedGradient2 = zeros([size(Theta2_global), N]);
    % training
    for k0 = 1 : K0
        % record cost function
        J = nnCostFunction(Theta1_global, Theta2_global, num_labels, img_train, lab_train, lambda);        %cost function
        J_step(k0,iter)=J;
        
        for n = 1 : N
            Theta1_local = Theta1_global;
            Theta2_local = Theta2_global;
            % local training
            for k = 1 : Kn(n)
                % sampling
                rand_idx = randperm(local_batch_size, B);
                X = local_img(:, rand_idx, n);
                y = local_lab(rand_idx, n);
                % local update
                [local_Grad1, local_Grad2] = nnGradient(Theta1_local,Theta2_local, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
                Theta1_local = Theta1_local - lr * local_Grad1;
                Theta2_local = Theta2_local - lr * local_Grad2;
            end
            [QuantizedGradient1(:, :, n), QuantizedGradient2(:, :, n)] = ...
                Quantization((Theta1_local - Theta1_global)/(lr*Kn(n)), (Theta2_local - Theta2_global)/(lr*Kn(n)), ...
                snt(n), sn(n), range);
        end
        % model update
        AverageGradient1 = sum(W(n)*Kn(n)*QuantizedGradient1, 3)/(sum(W.*Kn));
        AverageGradient2 = sum(W(n)*Kn(n)*QuantizedGradient2, 3)/(sum(W.*Kn));
        [QuantizedAverage1, QuantizedAverage2] = ...
            Quantization(AverageGradient1, AverageGradient2, s0t, s0, range);
        Theta1_global = Theta1_global + lr*QuantizedAverage1;
        Theta2_global = Theta2_global + lr*QuantizedAverage2;
        pred = Predict(Theta1_global, Theta2_global, img_test);
        acc(k0,iter)=mean(double(pred == lab_test)) * 100;
        fprintf('Test accuracy of step %d: %f\n', k0, mean(double(pred == lab_test)) * 100);
    end
    fprintf('\nComplete training of iteration %d\n', iter);
end
accuracy = mean(acc,2);
cost = mean(J_step,2);

