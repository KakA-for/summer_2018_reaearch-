function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%



% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network




neuron = [input_layer_size hidden_layer_size num_labels];

count = 1;

Theta = cell(1,length(neuron)-1);


 for i = 1:length(neuron)-1
        volume = neuron(i+1)*(neuron(i)+1);
        Theta{i} = reshape(nn_params(count:count+volume-1),neuron(i+1),neuron(i)+1);       
        count = count+volume;
 end



% 
% % raw
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


% Setup some useful variables
m = size(X, 1);
     
% You need to return the following variables correctly 
J = 0;

Theta_grad = cell(1,length(neuron)-1);

for i = 1:length(neuron)-1
    Theta_grad{i} = zeros(size(Theta{i}));
end

% % raw
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));
% Theta3_grad = zeros(size(Theta3));



% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X =[ones(m,1),X];

a = cell(1,length(neuron));

for i = 2:length(neuron)
    if i == 2
        a{i} = [ones(m,1) tanh_sigmoid(X*Theta{i-1}')];
    elseif i == length(neuron)
        a{i} = a{i-1}* Theta{i-1}';
    else
        a{i} = [ones(m,1) tanh_sigmoid(a{i-1}*Theta{i-1}')];
    end
end


% a2 = tanh_sigmoid(X *Theta1');
% a2 = [ones(m,1) a2];
% a3 = tanh_sigmoid(a2*Theta2');
% a3 = [ones(m,1) a3];
% a4 = a3* Theta3';





J = J+ sum(sum((y-a{length(neuron)}).^2))/m;

for i = 1:length(neuron)-1
    J = J+sum(sum(Theta{i}(:,2:end).^2))*lambda/(2*m);
end

% for i = 1:length(neuron)-1
%     J = J+sum(sum(Theta{i}(:,2:end).^2))*lambda/(2*m);
% end

% J  =J+ sum((y-a4).^2)/m;
% J =J+ (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) ...
%      +sum(sum(Theta3(:,2:end).^2)))*lambda/(2*m);

               

delta = cell(1,length(neuron));
for i = 1:length(neuron)-1
    delta{i} = zeros(size(Theta{i}));
end
  
  
% delta_1 = zeros(size(Theta1));
% delta_2 = zeros(size(Theta2));
% delta_3 = zeros(size(Theta3));



a   = cell(1,length(neuron));
z   = cell(1,length(neuron));
err = cell(1,length(neuron));
for t = 1:m
    yt = y(t);
    a{1} = X(t,:)';
    for i = 2:length(neuron)
        if i == length(neuron)
            z{i} = Theta{i-1}*a{i-1};
            a{i} = z{i};
        else
            z{i} = Theta{i-1}*a{i-1};
            a{i} = [1;tanh_sigmoid(z{i})];
        end    
    end
    
    for i = length(neuron):-1:2
        if i == length(neuron)
            err{i} = a{i} - yt;
            
        else
            err{i} = Theta{i}(:,2:end)'*err{i+1}.*tanh_sigmoidGradient(z{i});
            
        end
        
    end
    for i = 1:length(neuron)-1
        delta{i} = delta{i}+err{i+1}*a{i}';
    end
    
    
end



% for t = 1:m
%     
%    	yt = y(t);
%       a1 = X(t,:)';
%   	z2 = Theta1*a1;
%       a2 = tanh_sigmoid(z2);
%       a2 = [1;a2];
%       z3 = Theta2*a2;
%       a3 = tanh_sigmoid(z3);
%       a3 = [1;a3];
%       z4 = Theta3*a3;
%       a4 = z4;
%           
% %%% here is the point         
%     err_4 = a4 - yt;
%     err_3 = Theta3(:,2:end)'*err_4.*tanh_sigmoidGradient(z3);
%     err_2 = Theta2(:,2:end)'*err_3.*tanh_sigmoidGradient(z2); % there is z2, nota2f
%                 
%                 
%     delta_1 = delta_1+ err_2*a1';
%     delta_2 = delta_2 +err_3*a2';
%     delta_3 = delta_3+ err_4*a3';
% end
%   




%%%this is a critical part. gradient is 0 if theta i is unchanged when training



for i = 1:length(neuron)-1
    aa = lambda*Theta{i}/m;
    Theta_grad{i} =  delta{i}/m;
    Theta_grad{i}(:,2:end) = Theta_grad{i}(:,2:end)+ aa(:,2:end);  
end

% 
%  Theta_grad{1}(:,:) = 0 ;
%  Theta_grad{2}(:,:) = 0 ;
%         
 
% Theta1_grad =  delta_1/m;
% 
% Theta2_grad =  delta_2/m;
%                 
% Theta3_grad =  delta_3/m;
% 
% 
% aa = lambda*Theta1/m;
%                
% Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+ aa(:,2:end);
% 
% aa =lambda*Theta2/m;
% 
% Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+aa(:,2:end);
% 
% aa = lambda*Theta3/m;
% 
% Theta3_grad(:,2:end) = Theta3_grad(:,2:end)+aa(:,2:end);






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [];
for i = 1:length(neuron)-1
    grad = [grad;Theta_grad{i}(:)];
end

grad = grad*2;

%   grad = [Theta1_grad(:) ; Theta2_grad(:);Theta3_grad(:);]*2;


end
