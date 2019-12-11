function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
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

temp2(2:n)=theta(2:n).*theta(2:n);
temp2(1)=0;

J2=(1/(2*m))*lambda*sum(temp2);%factor for regularization 
J1=(-1/m)*sum(temp1);
J=J1+J2% total cost of all training set
temp2=0;
temp1=0;
%now calculating grad using vectorised implementation
%calculate length of theta explicitly inside each case 
%donot use previously calculated size of theta
z=X*theta;
z=z.*-1;
predict=1./(1+e.^z); %g(z)
  temp5=X'*(predict-y);
  temp6=zeros(size(theta));
  temp6(2:(length(theta)))=lambda.*theta(2:length(theta));
  
  grad=(1/m)*(temp5.+temp6);
  






% =============================================================

end
