% Your code here.
clear
close all
dir_name = '../images/';
file_names = dir([dir_name '*.jpg']);
for i= 1:length(file_names)
    file_path_name = [file_names(i).folder '/' file_names(i).name];
    im = imread(file_path_name);
  
    [lines, bw] = findLetters(im);
    figure;
    imshow(bw);
    hold on;
    for j = 1: length(lines)
        for l = 1: size(lines{j},1) %loop over letters
            x1 = lines{j}(l,1);
            y1 = lines{j}(l,2);
            x2 = lines{j}(l,3);
            y2 = lines{j}(l,4);
            rec_h = x2 - x1;
            rec_w = y2 - y1;   
            rectangle('Position', [x1, y1, rec_h,rec_w  ], 'LineWidth', 3, 'EdgeColor', 'r' );
            
        end
    end
    %Save image
    title(file_path_name);
    name = strcat('../results/q4_3_det_char_', num2str(i));
    name = strcat(name, '.jpg');
    Image = getframe(gcf);
    imwrite(Image.cdata, name);

    
end

