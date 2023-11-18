function [ convergence, K, K_0, B, gamma, W, s, s_0, ts, ts_0 ] = ConvergenceOptimization( N, D, alpha, ...
    C_0, F_0, p_0, r_0, F_n, C_n, p_n, r_n, delta, delta_0, ...
    Q, L, sigma, E_max, T_max, rand_init )

convergence = Inf;

c_1 = 2*Q;
c_2 = L^2*sigma^2/2;
c_3 = L*sigma^2;
c_4 = L;

%% =============== Solution: Proposed =================
% ------------- Initialize --------------
K_t = zeros(N, 1);
K_0_t = 0;
B_t = 0;
gamma_t = 0;
W_t = ones(N,1)/N;
ts_t = ones(N,1);
s_t = ones(N,1);
ts_0_t = 1;
s_0_t = 1;

K_init_list = [ 10:200, 250:50:1000 ];
K_0_init_list = [ 100:50:5000 ];
B_init_list = [ 1:20, 25:5:100 ];
gamma_init_list = [0.005:0.005:0.2];
W = rand_init;  W = W/sum(W);
ts = 10*(rand_init+5);
s = 10*(rand_init+5);
ts_0 = 10;
s_0 = 10;
S = log2(ts+1)+D*log2(s+1)+D;
S_0 = log2(ts_0+1)+D*log2(s_0+1)+D;
T_2 = max(S./r_n);
q = min([D./(s.^2), sqrt(D)./s], [], 2);
q_0 = min([D/(s_0^2), sqrt(D)/s_0]);
y = abs(D./(s.^2)-sqrt(D)./s);
y_0 = abs(D/(s_0^2)-sqrt(D)/s_0);

% ---------- Check -------------
is_feasible = 0;
for i1 = 1 : length(K_init_list)
    Rand = rand_init+5;  Rand = Rand/sum(Rand);
    K = K_init_list(i1)*Rand;
    T_1 = max(C_n./F_n.*K);
    for i2 = 1 : length(K_0_init_list)
        K_0 = K_0_init_list(i2);
        for i3 = 1 : length(B_init_list)
            B = B_init_list(i3);
            for i4 = 1 : length(gamma_init_list)
                gamma = gamma_init_list(i4);
                
                cons1 = isempty(   find( ((B*T_1+C_0/F_0+max(S./r_n)+S_0/r_0)*K_0 <= T_max)==0, 1 )   );      % Eq. 1
                cons2 = isempty(   find( ((B*sum(alpha.*C_n.*F_n.^2.*K)+alpha*C_0*F_0^2+sum(p_n.*S./r_n)+p_0*S_0/r_0)*K_0<=E_max)==0, 1 )   );      % Eq. 2
                
                if cons1 && cons2
                    is_feasible = 1;    fprintf('Feasible initial point found!\n');   break;
                end
                
                if is_feasible
                    break;
                end
            end
            if is_feasible
                break;
            end
        end
        if is_feasible
            break;
        end
    end
    if is_feasible
        break;
    end
end
if is_feasible == 0
    fprintf('Error: Feasible point not found!  \n');
    pause(1);
    %         continue;
    return;
end

% ------------- Algorithm --------------
conv_rcd = [];
norm_rcd = [];
cnt = 1;
while (norm([K_t-K])>1 || norm([K_0_t-K_0])>1 || norm([B_t-B])>1 || norm([gamma_t-gamma])>0.01 || norm([W_t-W])>0.01 ...
        || norm([ts_t-ts])/norm(ts)>0.01 || norm([s_t-s])/norm(s)>0.01 || norm([ts_0_t-ts_0])/norm(ts_0)>0.01 ...
        || norm([s_0_t-s_0])/norm(s_0)>0.01 ) && cnt < 200
    K_t = K;  K_0_t = K_0;  B_t = B;  gamma_t = gamma;  W_t = W;
    ts_t = ts;  s_t = s;  ts_0_t = ts_0;  s_0_t = s_0;  q_t = q;  q_0_t = q_0;
    y_t = y;  y_0_t = y_0;
    exp0 = W_t/sum(W_t);
    
    dem1 = q_t+y_t;  exp1_1 = 1e14*q_t./(1e14*dem1);  exp1_2 = 1e14*y_t./(1e14*dem1);
    dem2 = q_0_t+y_0_t;  exp2_1 = 1e14*q_0_t/(1e14*dem2);  exp2_2 = 1e14*y_0_t/(1e14*dem2);
    dem3 = y_t.*s_t.^2+D;  exp3 = y_t.*s_t.^2./(dem3);
    dem4 = y_0_t*s_0_t^2+D;  exp4 = y_0_t*s_0_t^2/(dem4);
    dem5 = y_t.*s_t+sqrt(D);  exp5 = 1e14*y_t.*s_t./(1e14*dem5);
    dem6 = y_0_t*s_0_t+sqrt(D);  exp6 = 1e14*y_0_t*s_0_t/(1e14*dem6);
    
    fprintf('Iteration %d ...\n', cnt);
    cvx_begin gp quiet
        cvx_solver SeDuMi
    variables K(N,1) K_0 B gamma W(N,1) ts(N,1) ts_0 s(N,1) s_0 S(N,1) S_0 q(N,1) q_0 T_1 T_2 T_3 T_4 y(N,1) y_0
    denominator = sum(W_t.*K_t) * prod((W.*K./(W_t.*K_t)).^(W_t.*K_t/sum(W_t.*K_t)));
    minimize (  c_1/( gamma*K_0*denominator ) ...
        + c_2*gamma^2*sum(W.*K.*(K+1))/(B*denominator) ...
        + c_3*gamma*(1+q_0)*sum((N+q).*W.^2.*K)/(B*denominator) ...
        + c_4*gamma*(1+q_0)*(delta_0^2/ts_0^2+sum((1+q).*W.^2.*K.^2.*delta.^2./(ts.^2)))/(4*denominator) ...
        + 100*sum(y) + 100*y_0  );
    subject to
    sum(W) <= 1;
    L^2*gamma^2*K+L*gamma*(1+q_0)*(N+q).*W.*K <= ones(N,1);
    (C_n./F_n.*K)/T_1 <= ones(N,1);  (S./r_n)/T_2 <= ones(N,1);
    q.*s.^2/D <= ones(N,1);  q_0*s_0/D <= 1;
    q.*s/sqrt(D) <= ones(N,1);  q_0*s_0/sqrt(D) <= 1;
    K_0*(B*T_1+C_0/F_0+T_2+S_0/r_0)/T_max <= 1;
    K_0*(B*sum(alpha*C_n.*F_n.^2.*K)+alpha*C_0*F_0^2+sum(p_n.*S./r_n)+p_0*S_0/r_0)/E_max <= 1;
    
    1/( sum(W_t)*prod((W./W_t).^(exp0)) ) <= 1;
    log2(ts_t+1)+(ts-ts_t)./((ts_t+1)*log(2))+D*log2(s_t+1)+D*(s-s_t)./((s_t+1)*log(2))+D <= S;
    log2(ts_0_t+1)+(ts_0-ts_0_t)/((ts_0_t+1)*log(2))+D*log2(s_0_t+1)+D*(s_0-s_0_t)/((s_0_t+1)*log(2))+D <= S_0;
    D./( dem1.*s.^2.*(q./q_t).^exp1_1.*(y./y_t).^exp1_2 ) <= ones(N,1);
    D/( dem2*s_0^2*(q_0/q_0_t)^exp2_1*(y_0/y_0_t)^exp2_2 ) <= 1;
    sqrt(D)./( dem1.*s.*(q./q_t).^exp1_1.*(y./y_t).^exp1_2 ) <= ones(N,1);
    sqrt(D)/( dem2*s_0*(q_0/q_0_t)^exp2_1*(y_0/y_0_t)^exp2_2 ) <= 1;
    sqrt(D)*s./( dem3.*(y.*s.^2./(y_t.*s_t.^2)).^exp3 ) <= ones(N,1);
    sqrt(D)*s_0/( dem4*(y_0*s_0^2/(y_0_t*s_0_t^2))^exp4 ) <= 1;
    D./( dem5.*s.*(y.*s./(y_t.*s_t)).^exp5 ) <= ones(N,1);
    D/( dem6*s_0*(y_0*s_0/(y_0_t*s_0_t))^exp6 ) <= 1;
    
    B <= 6000;  1/B <= 1;
    ts <= 2^32*ones(N,1);  s <= 2^32*ones(N,1);
    ts_0 <= 2^32;  s_0 <= 2^32;
    1./K <= ones(N,1);
    1./q <= 1/min([D./(2^64), sqrt(D)./(2^32)])*ones(N,1);  1/q_0 <= 1/min([D/(2^64), sqrt(D)/2^32]);
    
    cvx_end
    
    conv_rcd = [conv_rcd, ...
        c_1/(gamma*K_0*sum(W.*K)) ...
        + c_2*gamma^2*sum(W.*K.*(K+1))/(B*sum(W.*K)) ...
        + c_3*gamma*(1+q_0)*sum((N+q).*W.^2.*K)/(B*sum(W.*K)) ...
        + c_4*gamma*(1+q_0)*(delta_0^2/ts_0^2+sum((1+q).*W.^2.*K.^2.*delta.^2./(ts.^2)))/(4*sum(W.*K))];
    norm_rcd = [norm_rcd, ...
        norm([K_t; K_0_t; B_t; gamma_t; W_t; ts_t; s_t; ts_0_t; s_0_t] - [K; K_0; B; gamma; W; ts; s; ts_0; s_0], 2) ];
    
      cnt = cnt + 1;
end

convergence = c_1/(gamma*K_0*sum(W.*K)) ...
    + c_2*gamma^2*sum(W.*K.*(K+1))/(B*sum(W.*K)) ...
    + c_3*gamma*(1+q_0)*sum((N+q).*W.^2.*K)/(B*sum(W.*K)) ...
    + c_4*gamma*(1+q_0)*(delta_0^2/ts_0^2+sum((1+q).*W.^2.*K.^2.*delta.^2./(ts.^2)))/(4*sum(W.*K));

K_0 = floor(K_0);  K = floor(K);
ts = floor(ts);  ts_0 = floor(ts_0);
s = floor(s);  s_0 = floor(s_0);

fprintf('\n');
