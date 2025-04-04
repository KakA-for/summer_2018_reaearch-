load ra_net
load input_tar_a 
load output_tar_a 

load T_tar

input = T_tar(:,[1 2]);
output = T_tar(:,3);
ra_net.divideParam.trainRatio=0.98;
ra_net.divideParam.testRatio=0.01;
ra_net.divideParam.valRatio=0.01;
ra_net.performParam.regularization = 0.12;
ra_net = train(ra_net,input',output');
result = ra_net(input_tar_a');
err = result'-output_tar_a;