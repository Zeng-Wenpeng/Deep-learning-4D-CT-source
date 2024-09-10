function img2_aligned = registerImages(a, b, slice_index)
    % 注释：此函数用于对两个三维数据集进行刚性配准。
    % 若旋转角度超过设定阈值，将返回原始未配准的图像。
    % 输入：
    %   a - 参考数据集
    %   b - 需要配准的数据集
    %   slice_index - 用于配准的切片索引
    % 输出：
    %   img2_aligned - 配准后的数据集或原始数据集

    % 选择切片
    slice1 = a(:, :, slice_index);  % 从参考数据集中选取切片
    slice2 = b(:, :, slice_index);  % 从需要配准的数据集中选取切片

    % 创建二值化图像，用于提取边缘信息
    binary_slice1 = double(slice1 > 0);
    binary_slice2 = double(slice2 > 0);

    % 配置刚性配准参数
    [optimizer, metric] = imregconfig('monomodal');

    % 执行刚性配准，获得变换参数
    tform = imregtform(binary_slice2, binary_slice1, 'rigid', optimizer, metric);

    % 提取旋转角度，变换矩阵tform中的旋转部分为2x2矩阵
    rotationMatrix = tform.T(1:2, 1:2);
    rotationAngle = atan2(rotationMatrix(2,1), rotationMatrix(1,1)) * (180 / pi);  % 弧度转度

    % 检查旋转角度是否超过阈值，例如10度
    if abs(rotationAngle) > 10
        img2_aligned = b;  % 旋转角度过大，返回原始数据
    else
        % 应用得到的变换参数到整个三维数据集
        img2_aligned = zeros(size(b), 'like', b);  % 初始化配准后的数据集
        for z = 1:size(b, 3)
            img2_aligned(:, :, z) = imwarp(b(:, :, z), tform, 'OutputView', imref2d(size(slice2)));
        end
    end
end