clear

load hyp;
load input_tar_a 
load output_tar_a 
N_half = floor(N/2);

% X = input_tar_a;
% result = zeros(size(output_tar_a,1),ceil(N/2));
% for t = N_half:N
%    
%     currentFile = sprintf('model%d.mat',t);
%     %save(currentFile,'Theta1', 'Theta2', 'Theta3','betat');
%     load(currentFile);
%     result(:,t-N_half+1) = betat *predict(Theta1, Theta2, Theta3, X) ;
% end
% err = sum(result,2)- output_tar_a;

X = linspace(0,pi,1000);

    currentFile = sprintf('model%d.mat',1);
    %save(currentFile,'Theta1', 'Theta2', 'Theta3','betat');
    load(currentFile);
    result = betat * predict(Theta1, Theta2, Theta3, X') ;
answer = 2*sin(4*X);
%err = sum(result,2)-answer ;