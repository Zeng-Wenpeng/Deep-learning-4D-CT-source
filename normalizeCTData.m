%%用于dcm数据的归一化
function normalizedMatrix = normalizeCTData(ctMatrix, lowerThreshold, percentile)
    % ctMatrix: 输入的三维CT图像矩阵
    % lowerThreshold: 背景和前景的分界阈值
    % percentile: 用于确定异常值上限的百分位数，如95或99

    % 提取非背景的有效数据
    ctMatrix = double(ctMatrix);
    validData = ctMatrix(ctMatrix > lowerThreshold);

    % 计算百分位数阈值，排除异常高值
    upperThreshold = prctile(validData, percentile);

    % 计算归一化所需的最大值和最小值
    minValue = min(validData);
    maxValue = prctile(validData, percentile);  % 使用同一个百分位值作为最大值

    % 初始化归一化矩阵，背景初始化为较小的非零值，如0.01
    normalizedMatrix = ones(size(ctMatrix)) * 0.01;

    % 归一化非背景数据，排除异常高值
    mask = ctMatrix > lowerThreshold & ctMatrix <= upperThreshold;
    normalizedMatrix(mask) = (ctMatrix(mask) - minValue) / (maxValue - minValue);

    % 异常高的值和背景值设为0
    normalizedMatrix(ctMatrix <= lowerThreshold | ctMatrix > upperThreshold) = 0;
end