%load test data and model for 26 chars
clear
close all
load('nist26_model.mat');
load('../data/nist26_test.mat', 'test_data', 'test_labels')

[outputs] = Classify(W, b, test_data);
D = size(outputs,1);
C = size(outputs,2);
conf_mat = zeros(C,C);

for i = 1:D
   [~,cor_idx] = max(test_labels, [], 2);
   [~, pred_idx] = max(outputs, [], 2);
end

for i = 1:D
    conf_mat(cor_idx(i), pred_idx(i)) = conf_mat(cor_idx(i), pred_idx(i)) + 1;    
end
imagesc(conf_mat);
title('Confusion Matrix Plot')
name = '../results/q3_1_4_conf_color';
name = strcat(name, '.jpg');
Image = getframe(gcf);
imwrite(Image.cdata, name);


figure;
conf_big = imresize(mat2gray(conf_mat), 12, 'nearest' );
imshow(conf_big);

title('Confusion Matrix Plot')
name = '../results/q3_1_4_conf_bw';
name = strcat(name, '.jpg');
Image = getframe(gcf);
imwrite(Image.cdata, name);

conf_mat