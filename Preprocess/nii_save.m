clc;
clear;
% 设定参考文件名
referenceName = 's100_ori.mat';

% 加载参考数据集
referenceLoaded = load(referenceName);
referenceData = referenceLoaded.(char(fieldnames(referenceLoaded)));

% 获取当前文件夹内所有.mat文件
files = dir('*.mat');

% 遍历所有文件，与参考数据进行配准
for i = 1:length(files)
    fileName = files(i).name;
    
    % 跳过参考文件
    if strcmp(fileName, referenceName)
        continue;
    end

    % 加载需要配准的数据
    dataLoaded = load(fileName);
    data = dataLoaded.(char(fieldnames(dataLoaded)));

    % 执行配准（确保已有registerImages脚本或相应的代码）
    slice_index = 200;  % 选择一个合适的切片索引
    alignedData = registerImages(referenceData, data, slice_index);

    % 转换配准后的数据类型到uint8（如果数据范围是0-1）
    alignedData = uint8(alignedData * 255);

    % 将配准后的数据保存为NIfTI格式（确保已安装NIfTI工具箱）
    nii = make_nii(alignedData);
    save_nii(nii, replace(fileName, '.mat', '.nii'));  % 注意文件扩展名的更改
end

% 转换参考数据类型到uint8并保存为NIfTI格式
referenceData = uint8(referenceData * 255);
refNii = make_nii(referenceData);
save_nii(refNii, replace(referenceName, '.mat', '.nii'));
