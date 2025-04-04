
load('input_source');
load('output_source');


X = input_source;
y = output_source;


input_layer_size = 2;

hidden_layer_size = [13 11];

num_labels = 1;


neuron = [input_layer_size hidden_layer_size num_labels];

Theta = cell(1,length(neuron)-1);





fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 20000);

for i = 1:length(neuron)-1
Theta{i} = randInitializeWeights(neuron(i),neuron(i+1));
end

% initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size(1));
% initial_Theta2 = randInitializeWeights(hidden_layer_size(1), hidden_layer_size(2));
% initial_Theta3 = randInitializeWeights(hidden_layer_size(2), num_labels);

% Unroll parameters
initial_nn_params = [];
for i = 1:length(neuron)-1
    initial_nn_params = [initial_nn_params; Theta{i}(:);];
    
end


% initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];



%  You should also try different values of lambda
lambda = 0.0003;

% Create "short hand" for the cost function to be minimized

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
checkNNGradients;
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params


count = 1;

 for i = 1:length(neuron)-1
        volume = neuron(i+1)*(neuron(i)+1);
        Theta{i} = reshape(nn_params(count:count+volume-1),neuron(i+1),neuron(i)+1);       
        count = count+volume;
 end

% Theta1 = reshape(nn_params(1:hidden_layer_size(1) * (input_layer_size + 1)), ...
%                  hidden_layer_size(1), (input_layer_size + 1));
% 
% Theta2 = reshape(nn_params(1+hidden_layer_size(1) * (input_layer_size + 1):...
%                            hidden_layer_size(1) * (input_layer_size + 1)+...
%                            ((hidden_layer_size(2) * (hidden_layer_size(1) + 1)))), ...
%                  hidden_layer_size(2), (hidden_layer_size(1) + 1));
% 
% 
% Theta3 = reshape(nn_params(end-(hidden_layer_size(2) + 1)*num_labels+1:end), ...
%                  num_labels, (hidden_layer_size(2) + 1));
% Theta{1} = Theta1;
% Theta{2} = Theta2;
% Theta{3} = Theta3;
fprintf('Program paused. Press enter to continue.\n');
pp = predict(Theta, X);

err = (y-pp)./y*100;
err = err(:);
%mse = sum(err.^2)/324;
%plot(abs(err(:)));
max_percent = max(abs(err))
mean_percent = mean(abs(err))
Theta_pretrained = Theta;
save pretrained Theta_pretrained neuron
%surf(linspace(-1,1,17),linspace(-1,1,17),reshape(pp,17,17)-reshape(output_source,17,17))