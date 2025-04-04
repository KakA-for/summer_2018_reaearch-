load ('ra_net');
load('input_0729');
load('output_0729');
load('index');

% take 9th payload
input_source = input(1:17^2,:);
input_source(:,1) = 1; % replace payload with bias 1
output_source = output(17^2*8+1:17^2*9);
output_source(145) = 32; % replace error value to correct valoe
                         % error is caused by numeriacal reason

%% peature pouint extraction
surf(linspace(-1,1,17),linspace(-1,1,17),reshape(output_source,17,17))                  
% index = [1 1
%     1 0.625
%     1 0 
%     1 -0.625
%     1 -1
%     0.5 1
%     0.5 0.5 
%     0.5 -0.125
%     0.5 -0.75
%     0.5 -1
%     0 1
%     0 0.625
%     0 0
%     0 -0.625
%     0 -1
%     -0.75 1
%     -0.75 0.875
%     -0.75 0.125
%     -0.75 -0.5
%     -0.75 -1
%     -1 1
%     -1 0.625
%     -1 0
%     -1 -0.625
%     -1 -1];

%  payload_d = 7 ;
% 
%  n = size(input_source,1);
%  m = size(index,1);
%  input_tar_a = input((payload_d-1)*17^2+1: payload_d*17^2,:);
%  output_tar_a = output((payload_d-1)*17^2+1: payload_d*17^2);
%  output_tar = zeros(m,1);
%  
% % 
%  for i = 1:m
%      for j = 1:n
%          if sum(index(i,:) == input_tar_a(j,[2 3]))==2
%              output_tar(i) = output_tar_a(j);
%          end
%      end
%  end                 
% input_tar = [ones(m,1) index]; 
% 
% 
% weight = ones(m+n,1)*1/(m+n);
% N = 50;
% x = [input_source
%     input_tar];
% y = [output_source
%     output_tar];
% 
% 
% for t = 1:N
%     %1
%     %2
%     dt = max(abs(y(n+1:n+m)-ra_net(x(n+1:n+m,:)')')); 
%     et = abs(y-ra_net(x')')/dt;
%     %3
%     epsoliont = sum(et(n+1:n+m).*weight(n+1:n+m))/sum(weight(n+1:n+m));
%     if epsoliont>=0.5
%         n = t-1;
%         break;
%     end
%     betat = epsoliont/(1-epsoliont);
%     beta0 = 1/(1+(2*log(n/N))^0.5);
%     weight(1:n) = weight(1:n).*beta0.^et(1:n);
%     weight(n+1:n+m) =  weight(n+1:n+m).*betat.^(-et(n+1:n+m));
%     weight = weight/sum(weight)
%     
%     
% end
% 
% err = ra_net(x')'-y;