function [J, grad] = gradient_working_costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta)
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z=X*theta;
z=z.*-1;
predict=1./(1+e.^z); %g(z)
temp1=(y'*log(predict)+(1-y')*log(1-predict));

temp2=theta.*theta;

J2=lambda*sum(temp2)%factor for regularization 
J1=(1/(2*m))*sum(temp1)
J=J1+J2% total cost of all training set
%now calculating grad using vectorised implementation
predict=1./(1+e.^z); %g(z)
  %shift_theta = theta(2:size(theta));
  %theta_reg = [0;shift_theta];
  temp5=X'*(predict-y);
  temp6=lambda*theta(2:size(theta));
  temp3=[0;temp6]
  
  grad=(1/m)*(temp5+temp3)






% =============================================================

end
