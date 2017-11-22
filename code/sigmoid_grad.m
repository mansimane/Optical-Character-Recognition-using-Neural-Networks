function [grad] = sigmoid_grad(x)

grad = x.*(1-x);

end