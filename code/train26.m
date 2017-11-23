clc
clear
close all
num_epoch = 50;
classes = 26;
layers = [32*32, 400, classes];
learning_rate = 0.01;

load('../data/nist26_train.mat', 'train_data', 'train_labels')
load('../data/nist26_test.mat', 'test_data', 'test_labels')
load('../data/nist26_valid.mat', 'valid_data', 'valid_labels')

[W, b] = InitializeNetwork(layers);

%Plot Weights
figure;
W_reshape = zeros(32,32,1,400);
%W1 original : 400 x 1024
for i= 1:layers(2)
    W_reshape(:,:,1,i) = mat2gray(reshape(W{1}(i,:),[32,32])); 
end
montage(W_reshape)   %M-by-N-by-1-by-K
title('Layer 1 Weight Visualization before training');
name = strcat('../results/q3_1_3_weights_before_train', num2str(learning_rate));
name = strcat(name, '.jpg');
Image = getframe(gcf);
imwrite(Image.cdata, name);


train_acc_lst = zeros(1,num_epoch);
valid_acc_lst = zeros(1,num_epoch);
test_acc_lst = zeros(1,num_epoch);

train_loss_lst = zeros(1,num_epoch);
valid_loss_lst = zeros(1,num_epoch);
test_loss_lst = zeros(1,num_epoch);

for j = 1:num_epoch
    [W, b] = Train(W, b, train_data, train_labels, learning_rate);

    [train_acc, train_loss] = ComputeAccuracyAndLoss(W, b, train_data, train_labels);
    [valid_acc, valid_loss] = ComputeAccuracyAndLoss(W, b, valid_data, valid_labels);
    [test_acc, test_loss] = ComputeAccuracyAndLoss(W, b, test_data, test_labels);

    train_acc_lst(j) = train_acc;
    valid_acc_lst(j) = valid_acc;
    train_loss_lst(j) = train_loss;
    valid_loss_lst(j) = valid_loss;

    fprintf('Epoch %d - accuracy: %.5f, %.5f , %.5f\t loss: %.5f, %.5f, %.5f \n', j, train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss);
end

%Plot Accuracy
figure;
plot(train_acc_lst)
hold on 
plot(valid_acc_lst)
xlabel('Epochs');
ylabel('Accuracy')
legend('Train Accuracy','Valid Accuracy')
str = strcat('Train vs Valid Accuracy, Learning Rate: ', num2str(learning_rate));
title(str)
name = strcat('../results/q3_1_1_acc', num2str(learning_rate));
name = strcat(name, '.jpg');
Image = getframe(gcf);
imwrite(Image.cdata, name);


%Plot Loss
figure;
plot(train_loss_lst)
hold on 
plot(valid_loss_lst)
xlabel('Epochs');
ylabel('Loss')
legend('Train Loss','Valid Loss')
str = strcat('Train vs Valid Loss, Learning Rate: ', num2str(learning_rate));
title(str)
name = strcat('../results/q3_1_1_loss', num2str(learning_rate));
name = strcat(name, '.jpg');
Image = getframe(gcf);
imwrite(Image.cdata, name);

%Plot Weights
figure;
W_reshape = zeros(32,32,1,400);     %W1 original : 400 x 1024
for i= 1:layers(2)
    W_reshape(:,:,1,i) = mat2gray(reshape(W{1}(i,:),[32,32])); 
end
montage(W_reshape)   %M-by-N-by-1-by-K
title('Layer 1 Weight visualization after Training');
name = strcat('../results/q3_1_3_weights_after_train', num2str(learning_rate));
name = strcat(name, '.jpg');
Image = getframe(gcf);
imwrite(Image.cdata, name);


save('nist26_model.mat', 'W', 'b')
