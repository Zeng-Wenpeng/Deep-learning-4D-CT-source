clc
clear
folderPath = 'G:\0607\100\7\modify';
CT_ori = readCTslices(folderPath);
%%
Threshold = -847;
CT_norm = normalizeCTData(CT_ori, Threshold, 100);
%%
[orig_x, orig_y, orig_z] = size(CT_norm);
pad_x = (540 - orig_x) / 2;
pad_y = (550 - orig_y) / 2;
pad_z = 478 - orig_z;
% 确保填充大小为正整数
pad_x_left = floor(pad_x);
pad_x_right = ceil(pad_x);
pad_y_top = floor(pad_y);
pad_y_bottom = ceil(pad_y);

% 使用 padarray 函数进行填充
A_padded = padarray(CT_norm, [pad_x_left, pad_y_top, 0], 0, 'pre');
A_padded = padarray(A_padded, [pad_x_right, pad_y_bottom, 0], 0, 'post');
A_padded = padarray(A_padded, [0, 0, pad_z], 0, 'post');
volumeViewer(A_padded);
%%
parts = strsplit(folderPath, '\');
namePart = strcat('s',parts{3}, '_', parts{4});  % 连接部分字符串

% 定义文件名和变量名
fileName = [namePart, '.mat'];
varName = namePart;
% 将CT_norm重命名为自定义的变量名
eval([varName ' = A_padded;']);

% 保存变量到.mat文件
save(fileName, varName);
