function [text] = extractImageText(fname)
% [text] = extractImageText(fname) loads the image specified by the path 'fname'
% and returns the next contained in the image as a string.
    
    text = '';
    im = imread(fname);
    alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    [lines, bw] = findLetters(im);  %Train images: black alphabets on white background, same as bw
    load('nist36_model.mat', 'W', 'b');
    im_gray = rgb2gray(im);
    for i = 1: length(lines)
        for l = 1: size(lines{i},1) %loop over letters
            x1 = lines{i}(l,1);
            y1 = lines{i}(l,2);
            x2 = lines{i}(l,3);
            y2 = lines{i}(l,4);
            rec_h = x2 - x1;
            rec_w = y2 - y1;   
            %rectangle('Position', [y1, x1, rec_w, rec_h ], 'LineWidth', 3, 'EdgeColor', 'r' );
            patch = bw(y1:y2,x1:x2);
            %Making square
            [h, w] = size(patch);
            high = max(h,w);
            d = max(high - h, high - w);
            patch = padarray(patch, [round((high - h)/2), round((high - w)/2)],1,'both' );
            
            %Making image centered by adding extra padding 
            patch = padarray(patch, [7, 7],1,'both' );           
            patch = imresize(patch,[32,32]);
            %imshow(patch);
            
            data = patch(:);
            [outputs] = Classify(W, b, data');
            [pred_val, pred_idx] = max(outputs, [], 2);
            letter = alphabets(pred_idx);
            text = strcat(text, letter);
        end
        text = strcat(text,'\n');
    end
    imshow(im);
    fname
    sprintf(text)
    

end
