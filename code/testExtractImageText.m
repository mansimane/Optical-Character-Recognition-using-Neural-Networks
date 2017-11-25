% Your code here.
clear
close all
dir_name = '../images/';
file_names = dir([dir_name '*.jpg']);


for i= 1:length(file_names)
    file_path_name = [file_names(i).folder '/' file_names(i).name];
    [text] = extractImageText(file_path_name);
    
end