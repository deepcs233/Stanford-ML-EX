function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

if all(size(z)==[1,1]),

g=1/(1+exp(-z));

else,

m=size(z,1);
n=size(z,2);
onee=ones(m,n);

g=onee./(onee+exp(-z));
end

% =============================================================

end
