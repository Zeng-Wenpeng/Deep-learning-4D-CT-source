#!/usr/bin/python3

import os
import argparse
import time
import numpy as np
import torch
import nibabel as nib
from visualise import visualise_result_test
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import SimpleITK as sitk

# parse the commandline
parser = argparse.ArgumentParser()
# data organization parameters
parser.add_argument('--img-test-list', default=r'F:\U-net-CT\code\Function\指定配对新_2.txt',
                    help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', default=False, help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--save-dir', default=r'F:\U-net-CT\CTReg\trainingrecords_visual_6',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
# model
parser.add_argument('--model_type', type=str, default='unet', help='unet')
# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=10,
                    help='frequency of model saves (default: 100)')

parser.add_argument('--load-model', default=r'F:\U-net-CT\CTReg\trainingrecords8\0020.pt', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=4,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
parser.add_argument('--cps_Bspline', type=int, default=2,
                    help='the control points image space')
args = parser.parse_args()

processed_path = r'F:\U-net-CT\code\Function\processed_data'

bidir = args.bidir

# load and prepare training data

test_files = vxm.py.utils.read_file_list(args.img_test_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)




# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

generator_test = vxm.generators_zeng.gen_test(
    test_files, batch_size=1, bidir=args.bidir, segs=False, shape=(128, 128, 128), add_feat_axis=add_feat_axis)
# 测试生成器，运行10
# generator = vxm.generators.volgen_ct_nomask(
#     vol_names=test_files,    # 这里需要提供 vol_names 参数的值，例如文件路径列表
#     shape=(128, 128, 128),  # 假设这是你想要的体积形状
#     batch_size=1,           # 使用批量大小为1
#     segs=None,              # 假设不加载分割数据
#     np_var='vol',           # 假设使用默认的体积变量名
#     pad_shape=None,         # 假设不进行填充
#     resize_factor=1,        # 假设不进行缩放
#     add_feat_axis=True      # 假设需要添加特征轴
# )

# # 测试生成器，运行10个循环
# for i in range(10):
#     generated_data = next(generator)
#     # 假设 ct_names 是输出元组的最后一个元素
#     ct_names = generated_data[-1]  # 获取生成的 ct_names
#     print(f"循环 {i+1} 生成的 CT 文件名：")
#     print(ct_names)
#     print("\n")

# extract shape from sampled inputs
inshape = (128, 128, 128)



# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
print('gpu数量：', nb_gpus)
device = 'cuda'
# print(os.environ['CUDA_VISIBLE_DEVICES'])

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.torch.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
        network_type=args.model_type,
        cps_Bspline=args.cps_Bspline
    )


def interpolater(img, org_shape):
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    img = torch.nn.functional.interpolate(
        img, size=org_shape, mode='trilinear',
        align_corners=True).numpy().squeeze(0).squeeze(0)
    return img

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.torch_similarity.modules.NormalizedCrossCorrelationLoss().forward
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

# training loops
visual_path = args.save_dir
os.makedirs(visual_path, exist_ok=True)

visual_3D_path = args.save_dir + '\\' + '3D'
os.makedirs(visual_3D_path, exist_ok=True)

print('开始测试\n')
'''
下面添加验证模式，在验证模式结束时需要 net.train()
'''
"ITK-SNAP"
model.eval()
for step in range(100):    # steps_per_epoch！=len(data)/batch_size 是个固定值， generator随机产生本轮训练的数据，而不是每次所有数据都参与训练
    print(step)
    # print('step:', step)
    step_start_time = time.time()
    # generate inputs (and true outputs) and convert them to tensors
    inputs, y_true, name = next(generator_test)    # inputs: [ct, mr], y_true: seg
    # y_true = y_true.astype(np.uint8)
    inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
    # y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
    y_true = [
        torch.from_numpy(d.astype(np.uint8)).to(device).float().permute(0, 5, 1, 2, 3, 4) if len(d.shape) == 6 else
        torch.from_numpy(d.astype(np.uint8)).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

    with torch.no_grad():
        y_pred = model(*inputs, atlas_mask=None, registration=True)
        # 增加mask计算部分
        # 正向变形场，拿来变形mask
        pos_flow = y_pred[1]
        neg_flow = -pos_flow  # 用于反向mask

        vol = inputs[0].cpu().detach().numpy()  # (2, 1, 192, 160, 192)
        vol = vol * 255 / (vol.max() - vol.min())
        atlas = inputs[1].cpu().detach().numpy()
        atlas = atlas * 255 / (atlas.max() - atlas.min())
        moved = y_pred[0].cpu().detach().numpy()

        save_3D_moved = moved[0, 0, :, :, :]


        img = nib.load(os.path.join(processed_path, name + '.nii'))
        org_shape = img.get_fdata().shape
        save_3D_moved = interpolater(save_3D_moved, org_shape)
        img_affine = img.affine
        nib.Nifti1Image(save_3D_moved, img_affine).to_filename(visual_3D_path + '\\' + name + '_moved.nii.gz')

        moved = moved * 255 / (moved.max() - moved.min())
        posflow_show = pos_flow.cpu().detach().numpy()
        error_before = atlas - vol  # (2, 1, 192, 160, 192)
        error_after = atlas - moved

        visual = {}
        visual['vol'] = vol
        visual['moved'] = moved
        visual['atlas'] = atlas
        visual['disp_pred'] = posflow_show

        img_affine = img.affine
        # nib.Nifti1Image(posflow_show[0, :, :, :, :], img_affine).to_filename(visual_3D_path + '\\' + name + '_dsp.nii.gz')

        # nib.Nifti1Image(posflow_show[0, :, :, :, :].transpose(1, 2, 3, 0), img_affine).to_filename(visual_3D_path + '\\' + name + '_dsp.nii')
        d_save = posflow_show[0, :, :, :, :].transpose(1, 2, 3, 0)

        savedImg = sitk.GetImageFromArray(d_save)
        sitk.WriteImage(savedImg, visual_3D_path + '\\' + name + '_dsp.nii')
        # nib.save(posflow_show[0, :, :, :, :].transpose(1, 2, 3, 0), visual_3D_path + '\\' + name + '_dsp.nii')
        val_fig = visualise_result_test(visual, axis=0, save_result_dir=visual_path, epoch=step, save_name=name)



