clc;clear all;

%% == Start ==
%%%%%%%%%%%% Constant %%%%%%%%%%
N = 10;
alpha = 2e-28;
F_0 = 3e9;
C_0 = 100;
p_0 = 20;
r_0 = 7.5e7;

D = 128*784;
Q = 2.75;  L = 0.6521;  sigma = 4.3023;

%% =========== Setting for Step Size Rules: Varying Convergence =============
%  ------- Setting for System Parameter ---------
F_mean = 1e9;
F_ratio = 10;
F_min = 2*F_mean/(1+F_ratio);
F_max = 2*F_ratio*F_mean/(1+F_ratio);
F_n = F_max*ones(floor(N/2), 1);
F_n = [F_n; F_min*ones(ceil(N/2), 1)];

C_mean = 100e6;
C_ratio = 1;
C_min = 2*C_mean/(1+C_ratio);
C_max = 2*C_ratio*C_mean/(1+C_ratio);
C_n = C_max*ones(floor(N/2), 1);
C_n = [C_n; C_min*ones(ceil(N/2), 1)];
delta = ones(N,1);
delta_0 = 1;

BW = 20e6/N;
h_mean = 1;
h_ratio = 1;
h_min = 2*h_mean/(1+h_ratio);
h_max = 2*h_ratio*h_mean/(1+h_ratio);
p_mean = 1.5;
p_n = p_mean*ones(N,1);
noise = 1;
r_rand1 = zeros(1000,1);  r_rand2 = zeros(1000,1);
for i = 1 : 1000
h_rayleigh1 = raylrnd(h_max^2);  h_rayleigh2 = raylrnd(h_min^2);
r_rand1(i) = BW*log2(1+p_mean.*h_rayleigh1/noise);  r_rand2(i) = BW*log2(1+p_mean.*h_rayleigh2/noise);
end
r_mean1 = mean(r_rand1);  r_mean2 = mean(r_rand2);
r_n = [r_mean1*ones(N/2,1); r_mean2*ones(N/2,1)];

%% ========= Simulation 1 ========
% ---------- Setting for Simulation --------
E_max = 200;
T_max = 50;
I = 10;
C_opt = 10;

fprintf('E_max=%d and T_max=%d start ...\n', E_max, T_max);
for itr = 1 : I
    rand_init = rand(N,1);
    fprintf('Simulation %d\n', itr);
    [C_tmp, K_tmp, K_0_tmp, B_tmp, gamma_tmp, W_tmp, s_tmp, s_0_tmp, ts_tmp, ts_0_tmp] = ...
        ConvergenceOptimization( N, D, alpha, C_0, F_0, p_0, r_0, F_n, C_n, p_n, r_n, delta, delta_0, ...
        Q, L, sigma, E_max, T_max, rand_init );
    if C_tmp < C_opt
        C_opt = C_tmp;
        K_opt = K_tmp;  K_0_opt=K_0_tmp;  B_opt=B_tmp;  gamma_opt=gamma_tmp;
        W_opt=W_tmp;  s_opt=s_tmp;  s_0_opt=s_0_tmp;  ts_opt=ts_tmp;  ts_0_opt=ts_0_tmp;
    end
    fprintf('convergence=%0.1f\n', C_opt);
end

