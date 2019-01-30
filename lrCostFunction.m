function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

%COST FUNCTION
h_theta = sigmoid(X * theta);
J = (1/m) * sum((-1)*y.*log(h_theta) - (1 - y).*log(1 - h_theta)) + (lambda/(2*m))*sum(theta(2:end).^2);

%GRADIENT
X_trans = transpose(X);
beta = h_theta - y;

grad = (1/m) * X_trans * beta + (lambda/m) * theta;
grad(1) = (1/m) * sum(X_trans(1, :) * beta);

grad = grad(:);

end
