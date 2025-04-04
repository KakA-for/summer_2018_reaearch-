%% sample code
clear ; close all; clc

load('ra_net');
load('input_0729');
load('output_0729');
load('index');

%% take 9th payload
input_source = input(1:17^2,:);
input_source = input_source(:,[2 3]); % replace payload with bias 1
output_source = output(17^2*8+1:17^2*9);
output_source(145) = 32; % replace error value to correct valoe
                         % error is caused by numeriacal reason

save input_source input_source
save output_source output_source
 payload_d = 7 ;

 n = size(output_source,1);
 m = size(index,1);
 
 input_tar_a = input((payload_d-1)*17^2+1: payload_d*17^2,[2 3]);
 output_tar_a = output((payload_d-1)*17^2+1: payload_d*17^2);
 output_tar = zeros(m,1);
save input_tar_a input_tar_a
save output_tar_a output_tar_a
 for i = 1:m
     for j = 1:n
         if sum(index(i,:) == input_tar_a(j,:))==2
             output_tar(i) = output_tar_a(j);
         end
     end
 end                 
input_tar = index; 
save input_tar input_tar
save output_tar output_tar
A = [input_tar ra_net(input_tar')' ones(m,1)];
B = output_tar;
H = inv(A'*A)*A'*B;
T_scr_1 = [input_source output_source ones(n,1)];
T_scr_prime = [input_source T_scr_1*H];
T_tar = [input_tar output_tar];
T = [T_scr_prime;T_tar];

%% initial training
% X = T;
% y = T(:,3);

t = linspace(0,pi,1000);
source = [t' sin(4*t)'];

t = linspace(0,pi,5);
target = [t' 2*sin(4*t)'];
T = [source;target];
X = T(:,1);
y = T(:,2);

input_layer_size  = 1;
hidden_layer_size = [5 5];
num_labels = 1;







% N = 50;
% weight_tot = zeros(m+n,N+1);
% weight = ones(m+n,1)*1/(m+n);
% weight_tot(:,1) = weight';



N = 50;
weight_tot = zeros(1050,N+1);
weight = ones(1050,1)*1/(1050);
weight_tot(:,1) = weight';










% for t = 1:N
%     %1
%     fprintf('\nTraining Neural Network... \n')
%     initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size(1));
%     initial_Theta2 = randInitializeWeights(hidden_layer_size(1), hidden_layer_size(2));
%     initial_Theta3 = randInitializeWeights(hidden_layer_size(2), num_labels);
%     % Unroll parameters
%     initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];
% 
%     options = optimset('MaxIter', 2000);
% 
% 
%     %  You should also try different values of lambda
%     lambda = 3;
% 
%     % Create "short hand" for the cost function to be minimized
% 
%     costFunction = @(p) nnCostFunction(p, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size, ...
%                                    num_labels, X, y, lambda,weight);
%     [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% 
%     % Obtain Theta1 and Theta2 back from nn_params
%     Theta1 = reshape(nn_params(1:hidden_layer_size(1) * (input_layer_size + 1)), ...
%                  hidden_layer_size(1), (input_layer_size + 1));
% 
%     Theta2 = reshape(nn_params(1+hidden_layer_size(1) * (input_layer_size + 1):...
%                            hidden_layer_size(1) * (input_layer_size + 1)+...
%                            ((hidden_layer_size(2) * (hidden_layer_size(1) + 1)))), ...
%                  hidden_layer_size(2), (hidden_layer_size(1) + 1));
% 
% 
%     Theta3 = reshape(nn_params(end-(hidden_layer_size(2) + 1)*num_labels+1:end), ...
%                  num_labels, (hidden_layer_size(2) + 1));
%     
%     pp = predict(Theta1, Theta2, Theta3, X);
%     %2
%     dt = max(abs(y(n+1:n+m)-pp(n+1:n+m))); 
%     et = abs(y-pp)/dt;
%     %3
%     epsoliont = sum(et(n+1:n+m).*weight(n+1:n+m)/sum(weight(n+1:n+m)));
%     betat = epsoliont/(1-epsoliont);
%     currentFile = sprintf('model%d.mat',t);
%     save(currentFile,'Theta1', 'Theta2', 'Theta3','betat');
%     if epsoliont>=0.5
%         N = t-1;
%         break;
%     end
%     
%     beta0 = 1/(1+(2*log(n/N))^0.5);
%     
%     weight(1:n) = weight(1:n).*beta0.^et(1:n);
%     weight(n+1:n+m) =  weight(n+1:n+m).*betat.^(-et(n+1:n+m));
%     
%     zm = sum(weight(1:n).*beta0.^et(1:n))+sum(weight(n+1:n+m).*betat.^(-et(n+1:n+m)));
%     weight = weight/zm;
%     weight_tot(:,t+1) = weight';
% 
% end
% save hyp N
% save weight_tot weight_tot





% 
% fprintf('\nTraining Neural Network... \n')
% 
% %  After you have completed the assignment, change the MaxIter to a larger
% %  value to see how more training helps.
% options = optimset('MaxIter', 500);
% 
%     initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size(1));
%     initial_Theta2 = randInitializeWeights(hidden_layer_size(1), hidden_layer_size(2));
%     initial_Theta3 = randInitializeWeights(hidden_layer_size(2), num_labels);
%     % Unroll parameters
%     initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];
% %  You should also try different values of lambda
% lambda = 3;
% 
% % Create "short hand" for the cost function to be minimized
% 
% costFunction = @(p) nnCostFunction(p, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size, ...
%                                    num_labels, X, y, lambda);
% 
% % Now, costFunction is a function that takes in only one argument (the
% % neural network parameters)
% 
% [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% 
% % Obtain Theta1 and Theta2 back from nn_params
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
% fprintf('Program paused. Press enter to continue.\n');
% pp = predict(Theta1, Theta2, Theta3, X);










%%

%https://www.hindawi.com/journals/tswj/2014/282747/
%https://github.com/LinZhineng/transfer-learning/tree/master/tradaboost