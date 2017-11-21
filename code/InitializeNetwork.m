function [W, b] = InitializeNetwork(layers)
% InitializeNetwork([INPUT, HIDDEN, OUTPUT]) initializes the weights and biases
% for a fully connected neural network with input data size INPUT, output data
% size OUTPUT, and HIDDEN number of hidden units.
% It should return the cell arrays 'W' and 'b' which contain the randomly
% initialized weights and biases for this neural network.

% Your code here

H = layers(2);  %Number of hidden layers

for i=1:len(layers)-1
    N = layers(i);
    H = layers(i+1);  %Number of hidden layers

    W{1} = normrnd(0,2.0/(H+N),[H,N]);
    b{1} = normrnd(0,1.0/(H),[H,1]);
    
end


C = size(b{end},1);
assert(size(W{1},2) == 1024, 'W{1} must be of size [H,N]');
assert(size(b{1},2) == 1, 'b{end} must be of size [H,1]');
assert(size(W{end},1) == C, 'W{end} must be of size [C,H]');

end
