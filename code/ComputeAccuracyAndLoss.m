function [accuracy, loss] = ComputeAccuracyAndLoss(W, b, data, labels)
% [accuracy, loss] = ComputeAccuracyAndLoss(W, b, X, Y) computes the networks
% classification accuracy and cross entropy loss with respect to the data samples
% and ground truth labels provided in 'data' and labels'. The function should return
% the overall accuracy and the average cross-entropy loss.
[~,N] = size(data);
[~,C] = size(labels);
assert(size(W{1},2) == N, 'W{1} must be of size [~,N]');
assert(size(b{1},2) == 1, 'b{1} must be of size [~,1]');
assert(size(b{end},2) == 1, 'b{end} must be of size [~,1]');
assert(size(W{end},1) == C, 'W{end} must be of size [C,~]');

%Your code here
[D,C] = size(labels);

[outputs] = Classify(W, b, data);
[pred_val, pred_idx] = max(outputs, [], 2);
[cor_val, cor_idx] = max(labels, [], 2);

%sub2ind(size(outputs), 1:length(cor_idx), cor_idx');
cor_prob = outputs(sub2ind(size(outputs), 1:length(cor_idx), cor_idx'));
loss = -sum(log(cor_prob));
loss = loss/D;
accuracy = sum(double(pred_idx == cor_idx))/D;

end