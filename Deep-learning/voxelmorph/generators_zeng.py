import numpy as np
import os
import itertools
import glob

def gen_train(vol_names, segs, shape=None, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training).
            Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    gen = gen_1(vol_names, shape, batch_size=batch_size, segs=segs, **kwargs)
    while True:
        ct_mr = next(gen)
        scan1 = ct_mr[0]
        scan2 = ct_mr[1][0]

        '''
        scan2scan也需要归一化，否则loss很大
        '''

        scan1 = scan1 / (scan1.max() - scan1.min())
        scan2 = scan2 / (scan2.max() - scan2.min())
        # print(scan1.max())

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [scan2, scan1]
        outvols = [scan2, scan1] if bidir else [scan1]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)

def gen_test(vol_names, segs, shape=None, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    """
    Generator for scan-to-scan registration.

    Parameters:
        vol_names: List of volume files to load, or list of preloaded volumes.
        bidir: Yield input image as output for bidirectional models. Default is False.
        batch_size: Batch size. Default is 1.
        prob_same: Induced probability that source and target inputs are the same. Default is 0.
        no_warp: Excludes null warp in output list if set to True (for affine training).
            Default if False.
        kwargs: Forwarded to the internal volgen generator.
    """
    zeros = None
    #gen = volgen_ct_nomask_test(vol_names, shape, batch_size=batch_size, segs=segs, **kwargs)
    gen = gen_1(vol_names, shape, batch_size=batch_size, segs=segs, **kwargs)    
    while True:
        ct_mr = next(gen)
        scan1 = ct_mr[0]
        scan2 = ct_mr[1][0]
        name = ct_mr[2]

        '''
        scan2scan也需要归一化，否则loss很大
        '''

        scan1 = scan1 / (scan1.max() - scan1.min())
        scan2 = scan2 / (scan2.max() - scan2.min())
        # print(scan1.max())

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [scan2, scan1]
        outvols = [scan2, scan1] if bidir else [scan1]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols, name)

def gen_1(
        vol_names,
        shape,
        batch_size=1,
        segs=None,
        np_var='vol',
        pad_shape=None,
        resize_factor=1,
        add_feat_axis=True
):
    """
    基于随机体积加载的生成器。体积可以通过目录路径、全局模式、文件路径列表或预加载体积列表传递。
    如果提供了 `segs`，则同时加载相应的分割，`segs` 可以是文件路径列表或预加载的分割，或者设置为 True。
    在 `segs` 为 True 的情况下，期望 npz 文件包含名为 'vol' 和 'seg' 的变量。支持传递预加载的体积（和可选的预加载分割）到生成器中。
    
    参数:
        vol_names: 体积文件加载路径、全局模式、文件路径列表或预加载体积列表。
        batch_size: 批量大小，默认为1。
        segs: 加载对应的分割，默认为None。
        np_var: 加载npz文件时使用的体积变量名，默认为'vol'。
        pad_shape: 加载体积后进行零填充至指定形状，默认为None。
        resize_factor: 体积调整大小的因子，默认为1。
        add_feat_axis: 加载体积数组时是否添加特征轴，默认为True。
    """

    # 转换全局路径为文件名
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    if isinstance(segs, list) and len(segs) != len(vol_names):
        raise ValueError('图像文件数量必须与分割文件数量匹配。')

    while True:
        # 生成随机图像索引
        indices = np.random.randint(len(vol_names), size=batch_size)

        # 加载体积并合并
        load_params = dict(shape=shape, np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis,
                           pad_shape=pad_shape, resize_factor=resize_factor)
        imgs_mr = [py.utils.load_volfile(vol_names[i], **load_params) for i in indices]
        vols_mr = [np.concatenate(imgs_mr, axis=0)]

        # 检查并加载CT图像
        imgs_ct = []
        for i in indices:
            while True:
                random_suffix = np.random.randint(1, 10)
                new_path = re.sub(r'_(\d+)(\.\w+)$', f'_{random_suffix}\\2', vol_names[i])
                if os.path.exists(new_path):
                    img_ct = py.utils.load_volfile(new_path, **load_params)
                    imgs_ct.append(img_ct)
                    break


        vols_ct = [np.concatenate(imgs_ct, axis=0)]

        vols_mr.append(vols_ct)
        vols_mr.append(vol_names[indices[0]].split('\\')[-1].split('.')[0])
        yield tuple(vols_mr)  # 这是一个生成器