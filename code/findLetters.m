function [lines, bw] = findLetters(im)
% [lines, BW] = findLetters(im) processes the input RGB image and returns a cell
% array 'lines' of located characters in the image, as well as a binary
% representation of the input image. The cell array 'lines' should contain one
% matrix entry for each line of text that appears in the image. Each matrix entry
% should have size Lx4, where L represents the number of letters in that line.
% Each row of the matrix should contain 4 numbers [x1, y1, x2, y2] representing
% the top-left and bottom-right position of each box. The boxes in one line should
% be sorted by x1 value.


%Your code here
img_gray = rgb2gray(im);
sigma = 3;
%bw_g = imgaussfilt(img_gray, sigma);
bw_bin = imcomplement(imbinarize(img_gray));
CC = bwconncomp(bw_bin);
num_objs = CC.NumObjects;
close all;
imshow(bw_bin);
hold on;
numPixels = cellfun(@numel,CC.PixelIdxList);


for i = 1:num_objs
    if numPixels(i) < 400
        continue
    end
        
    [row,col] = ind2sub(size(bw_bin), CC.PixelIdxList{i});
    x1 = min(row);
    y1 = min(col);
    x2 = max(row);
    y2 = max(col);
    
    fprintf('%d\t %d\t %d\t %d\n ',x1,y1,x2,y2)
    rec_h = x2 - x1;
    rec_w = y2 - y1;    
    rectangle('Position', [y1, x1, rec_w, rec_h ], 'LineWidth', 4, 'EdgeColor', 'r' );
    
    
    
end



assert(size(lines{1},2) == 4,'each matrix entry should have size Lx4');
assert(size(lines{end},2) == 4,'each matrix entry should have size Lx4');
lineSortcheck = lines{1};
assert(issorted(lineSortcheck(:,1)) | issorted(lineSortcheck(end:-1:1,1)),'Matrix should be sorted in x1');

end
