function [ Theta1_out, Theta2_out ] = Quantization( Theta1_in, Theta2_in, st, s, range )

input_vector = [Theta1_in(:); Theta2_in(:)];
input_norm = norm(input_vector, 2);
norm_floor = min(floor( st*input_norm/range ), st);
if rand < st*input_norm/range - norm_floor
    norm_level = (norm_floor+1)/st;
else
    norm_level = norm_floor/st;
end

vector_sign = sign(input_vector);
vector_floor = min(floor( s*abs(input_vector)/input_norm ), s);
vector_prob = s*abs(input_vector)/input_norm - vector_floor;

rand_vec = rand(length(input_vector), 1);
zero_one = (rand_vec < vector_prob);
vector_level = (vector_floor + zero_one)/s;

output_vector = range * norm_level * vector_sign .* vector_level;
Theta1_out = reshape(output_vector(1: length(Theta1_in(:))), size(Theta1_in));
Theta2_out = reshape(output_vector(length(Theta1_in(:))+1: end), size(Theta2_in));
end

