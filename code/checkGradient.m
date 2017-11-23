% Your code here.


num_epoch = 30;
classes = 26;
layers = [32*32, 400, classes];
X = randi([1,10],1024 , 1);
Y = zeros(26,1);
Y(4) = 1;
[W, b] = InitializeNetwork(layers);
[output, act_h, act_a] = Forward(W, b, X);
[grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a);
e = 1e-4;
% num_grad_W = grad_W;
% num_grad_b = grad_b;
d_th = 0.0001;
% for i=1:length(grad_W)
%     num_grad_W{i} = num_grad_W{i}*0;
%     num_grad_b{i} = num_grad_b{i}*0;
%    
% end

for i=1:length(W)
    W_copy = W;
    b_copy = b;
    for j=1:numel(W{i})  
        W_p = W;
        W_n = W;
        perturb = zeros(size(W{i}));
        perturb(j) = e;
        W_p{i} = W{i} + perturb;
        W_n{i} = W{i} - perturb;
        [~, loss_p] = ComputeAccuracyAndLoss(W_p, b, X', Y');
        [~, loss_n] = ComputeAccuracyAndLoss(W_n, b, X', Y');
        g = (loss_p - loss_n)/(2*e);
        %num_grad_W{i}(j) = g;
        diff = abs(grad_W{i}(j) - g);
        if (diff > d_th)
            fprintf('Gradient Calculation wrong in layer %d for W\n', i); 
        end

    end
    
    %%% Gradient calculation for biases
    for j=1:numel(b{i})  
        b_p = b;
        b_n = b;
        perturb = zeros(size(b{i}));
        perturb(j) = e;
        b_p{i} = b{i} + perturb;
        b_n{i} = b{i} - perturb;
        [~, loss_p] = ComputeAccuracyAndLoss(W, b_p, X', Y');
        [~, loss_n] = ComputeAccuracyAndLoss(W, b_n, X', Y');
        g = (loss_p - loss_n)/(2*e);
        %num_grad_b{i}(j) = g;
        diff = abs(grad_b{i}(j) - g);
        if (diff > d_th)
            fprintf('Gradient Calculation wrong in layer %d for b\n', i); 
        end    
    end
    
    
end
