function p = predict(Theta, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);


% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

temp = tanh_sigmoid([ones(m, 1) X] * Theta{1}');

for i = 2:length(Theta)-1 
    
    temp = tanh_sigmoid([ones(m, 1) temp] * Theta{i}');
end


p = [ones(m, 1) temp] * Theta{end}';


% =========================================================================


end
