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
sigma = 2;
bw_g = imgaussfilt(img_gray, sigma);
bw = imbinarize(bw_g);
bw_bin = imcomplement(bw);
CC = bwconncomp(bw_bin);
num_objs = CC.NumObjects;
% close all;
% imshow(bw_bin);
% hold on;

numPixels = cellfun(@numel,CC.PixelIdxList);

lines = {};
letters = [];
for i = 1:num_objs
    if numPixels(i) < 600
        continue
    end
        
    [row,col] = ind2sub(size(bw_bin), CC.PixelIdxList{i});
    x1 = min(row);
    y1 = min(col);
    x2 = max(row);
    y2 = max(col);
    
    %fprintf('%d\t %d\t %d\t %d\n ',x1,y1,x2,y2)
    %rec_h = x2 - x1;
    %rec_w = y2 - y1;    
    %rectangle('Position', [y1, x1, rec_w, rec_h ], 'LineWidth', 4, 'EdgeColor', 'r' );
    letters = [letters; [x1, y1, x2, y2]];
    
end

[~,idx] = sort(letters(:,1)); % sort just the first column
sorted_letters = letters(idx,:);   % sort the whole matrix using the sort indices
for i = 1:size(sorted_letters,1)
    if length(lines) == 0
        lines{1} = sorted_letters(i,:);
        j =1;
    else
        x1 = sorted_letters(i,1);
        if (x1 >= lines{j}(1,1)) && (x1 <= lines{j}(1,3))
            %Append in existing line
            lines{j} = [lines{j}; sorted_letters(i,:)];
        else
            %New lines
            j = j + 1;
            lines{j} = sorted_letters(i,:);
        end
    end
end


assert(size(lines{1},2) == 4,'each matrix entry should have size Lx4');
assert(size(lines{end},2) == 4,'each matrix entry should have size Lx4');
lineSortcheck = lines{1};
assert(issorted(lineSortcheck(:,1)) | issorted(lineSortcheck(end:-1:1,1)),'Matrix should be sorted in x1');

end
