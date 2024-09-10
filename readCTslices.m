%%本函数用于读取dcm切片生成三维矩阵
function ctMatrix = readCTslices(folderPath)
    % 读取指定文件夹下所有的.dcm文件
    files = dir(fullfile(folderPath, '*.dcm'));

    % 检查是否有文件被找到
    if isempty(files)
        error('没有找到.dcm文件，请检查文件夹路径');
    end

    % 读取第一张DICOM图像以获取尺寸信息
    sampleImage = dicomread(fullfile(folderPath, files(1).name));
    [height, width] = size(sampleImage);

    % 初始化三维矩阵
    numFiles = length(files);
    ctMatrix = zeros(height, width, numFiles, 'like', sampleImage);

    % 读取所有DICOM图像并填充到三维矩阵中
    for k = 1:numFiles
        fileName = fullfile(folderPath, files(k).name);
        ctMatrix(:, :, k) = dicomread(fileName);
    end
end