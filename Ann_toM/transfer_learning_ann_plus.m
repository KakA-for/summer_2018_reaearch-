clear all;
clc;


load('input_0729');
load('output_0729_corrected');
% load('input_tar');
% load('output_tar');
load('pretrained');

% output_tar_a(145) = 30.5652;
% output_tar(13) = 30.5652;


X = input;
y = output;
 




    
    

X_trans =[X(:,1) tanh(predict(Theta_pretrained(1:length(Theta_pretrained)-1),X(:,[2 3])))];
num_labels = 1;



test_sample = [2 4 6 8 10 12 14 16];
test_num = length(test_sample);
id = 0;
count = [0 0];
X_test = zeros(17^2*test_num,size(X_trans,2));
X_train = zeros(17^2*(17-test_num),size(X_trans,2));
 
y_test = zeros(17^2*test_num,size(y,2));
y_train = zeros(17^2*(17-test_num),size(y,2));
for i = 1:17
    
    if count(1,2)+1<= test_num
         if test_sample(count(1,2)+1) == i
          id = 1;
        else
          id = 0;
        end
    else
        id = 0;
    end
   
    
    
    if id == 1
        X_test(17^2*count(1,2)+1:17^2*(count(1,2)+1),:) = X_trans(17^2*(i-1)+1:17^2*i,:);
        y_test(17^2*count(1,2)+1:17^2*(count(1,2)+1),:) = y(17^2*(i-1)+1:17^2*i,:);
        count(1,2) = count(1,2)+1;
    else
        X_train(17^2*count(1,1)+1:17^2*(count(1,1)+1),:) = X_trans(17^2*(i-1)+1:17^2*i,:);
        y_train(17^2*count(1,1)+1:17^2*(count(1,1)+1),:) = y(17^2*(i-1)+1:17^2*i,:);
        count(1,1) = count(1,1)+1;
    end
end
    
count


%% just adjust here

added_layer_size = 25;
lambda = 0.1;
%%

hidden_layer_size = added_layer_size;

input_layer_size = size(X_trans,2);
neuron = [input_layer_size hidden_layer_size num_labels];
Theta = cell(1,length(neuron)-1);
% 
% Theta{1} = Theta1;
% Theta{2} = Theta2;

fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 100);

for i = 1:length(neuron)-1
Theta{i} = randInitializeWeights(neuron(i),neuron(i+1));
end



% Unroll parameters
initial_nn_params = [];
for i = 1:length(neuron)-1
    initial_nn_params = [initial_nn_params; Theta{i}(:);];   
end

%  You should also try different values of lambda



% Create "short hand" for the cost function to be minimized

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_trans, y, lambda);
                               
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

count = 1;
 for i = 1:length(neuron)-1
        volume = neuron(i+1)*(neuron(i)+1);
        Theta{i} = reshape(nn_params(count:count+volume-1),neuron(i+1),neuron(i)+1);       
        count = count+volume;
 end
 
% Theta = [Theta_pretrained(1,1:end-1) Theta];
fprintf('Program paused. Press enter to continue.\n');
X = X_train;
y = y_train;
pp = predict(Theta, X);

err_train = (y-pp)./y*100;
err_train = err_train(:);
%mse = sum(err.^2)/324;
%plot(abs(err(:)));
max_percent = max(abs(err_train))
mean_percent = mean(abs(err_train))
figure()
plot(abs(err_train(:)));
fprintf('actual');
X = X_test;
y = y_test;
                               
pp = predict(Theta, X);

err_test = (y-pp)./y*100;
err_test = err_test(:);
%mse = sum(err.^2)/324;
figure()
plot(abs(err_test(:)));
max_percent = max(abs(err_test))
mean_percent = mean(abs(err_test))                            
