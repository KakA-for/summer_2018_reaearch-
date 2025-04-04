function g = tanh_sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0.*(exp(z) - exp(-z))./(exp(z) + exp(-z));
end
