import os
import argparse
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from visualise import visualise_result
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
import random

# parse the commandline
parser = argparse.ArgumentParser()
# data organization parameters
parser.add_argument('--img-list', default=r'G:\U-net-CT\code\Function\处理后的文本.txt',
                    help='line-seperated list of training files')
parser.add_argument('--img-test-list', default=r'G:\U-net-CT\code\Function\新建 文本文档.txt',
                    help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', default=False, help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default=r'G:\U-net-CT\CTReg\trainingrecords_ssim_256_623',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
# model
parser.add_argument('--model_type', type=str, default='unet', help='unet')
# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=10,
                    help='frequency of model saves (default: 100)')

parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 32, 64, 128, 256)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 256, 128, 64, 64, 32, 32, 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=4,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
# loss hyperparameters
parser.add_argument('--image-loss', default='ssim',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.0001,
                    help='weight of deformation loss (default: 0.01)')
parser.add_argument('--cps_Bspline', type=int, default=2,
                    help='the control points image space')
args = parser.parse_args()


bidir = args.bidir

# load and prepare training data
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)

test_files = vxm.py.utils.read_file_list(args.img_test_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)


assert len(train_files) > 0, 'Could not find any training data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel


generator = vxm.generators_zeng.gen_train(
    train_files, batch_size=args.batch_size, bidir=args.bidir, segs=False, shape=(256, 256, 256),
    add_feat_axis=add_feat_axis)
generator_test = vxm.generators_zeng.gen_train(
    test_files, batch_size=1, bidir=args.bidir, segs=False, shape=(256, 256, 256),add_feat_axis=add_feat_axis)

# extract shape from sampled inputs
inshape = (256, 256, 256)


# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
tb_writer = SummaryWriter(model_dir)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
print('gpu数量：', nb_gpus)
device = 'cuda'
# print(os.environ['CUDA_VISIBLE_DEVICES'])

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 64]
dec_nf = args.dec if args.dec else [64, 32, 32, 32, 32, 16, 16]

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
elif args.image_loss == 'ssim':
    image_loss_func = vxm.losses.SSIM3D().loss
elif args.image_loss == 'tidu':
    image_loss_func = vxm.losses.GradientLoss3D().loss
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
min_dice_loss = 20.0
visual_path = os.path.join(model_dir, 'visual')
os.makedirs(visual_path, exist_ok=True)


print('开始训练\n')
for epoch in range(args.initial_epoch, args.epochs):
    print('epoch:', epoch)

    # save model checkpoint
    if epoch % 1 == 0 and epoch != 0 :
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(150):    # steps_per_epoch！=len(data)/batch_size 是个固定值， generator随机产生本轮训练的数据，而不是每次所有数据都参与训练
        # print('step:', step)
        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)    # inputs: [ct, mr], y_true: seg
        # y_true = y_true.astype(np.uint8)
        # inputs = [torch.from_numpy(d).to(device).float().unsqueeze(0).permute(0, 4, 1, 2, 3) for d in inputs]
        # # y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
        # y_true = [
        #     torch.from_numpy(d.astype(np.uint8)).to(device).float().unsqueeze(0).permute(0, 5, 1, 2, 3, 4) if len(d.shape) == 6 else
        #     torch.from_numpy(d.astype(np.uint8)).to(device).float().unsqueeze(0).permute(0, 4, 1, 2, 3) for d in y_true]

        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
        # y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
        y_true = [
            torch.from_numpy(d.astype(np.uint8)).to(device).float().permute(0, 5, 1, 2, 3, 4) if len(d.shape) == 6 else
            torch.from_numpy(d.astype(np.uint8)).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]


        y_pred = model(*inputs, atlas_mask=None, registration=True)

        pos_flow = y_pred[1]
        neg_flow = -pos_flow

        if epoch % 1 == 0 and step < 1:
            vol = inputs[0].cpu().detach().numpy()      # (2, 1, 192, 160, 192)
            vol = vol * 255 / (vol.max() - vol.min())
            atlas = inputs[1].cpu().detach().numpy()
            atlas = atlas * 255 / (atlas.max() - atlas.min())
            moved = y_pred[0].cpu().detach().numpy()
            moved = moved * 255 / (moved.max() - moved.min())
            posflow_show = pos_flow.cpu().detach().numpy()
            error_before = atlas - vol      # (2, 1, 192, 160, 192)
            error_after = atlas - moved

            visual = {}
            visual['vol'] = vol
            visual['moved'] = moved
            visual['atlas'] = atlas
            visual['disp_pred'] = posflow_show
            val_fig = visualise_result(visual, axis=2, save_result_dir=visual_path, epoch=epoch)
            tb_writer.add_figure('visual', val_fig, global_step=epoch)

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            if n == 0:
                '''
                inputs[1] 这里指的是 atlas
                '''
                curr_loss = loss_function(inputs[1], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss
            else:
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss
        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())
        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # get compute time
        epoch_step_time.append(time.time() - step_start_time)


    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    tb_writer.add_scalar('registration_loss', np.mean(epoch_loss, axis=0)[0], epoch + 1)
    tb_writer.add_scalar('smoothing_loss', np.mean(epoch_loss, axis=0)[1], epoch + 1)
    tb_writer.add_scalar('epoch_total_loss', np.mean(epoch_total_loss, axis=0), epoch + 1)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
    # 在每个epoch结束时，使用测试集进行评估
    model.eval()
    with torch.no_grad():
        test_loss = 0
        num_samples = 30  # 你可以根据需要调整这个值
        
        for _ in range(num_samples):
            inputs, y_true = next(generator_test)
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d.astype(np.uint8)).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

            y_pred = model(*inputs, atlas_mask=None, registration=True)
            loss = 0
            for n, loss_function in enumerate(losses):
                if n == 0:
                    curr_loss = loss_function(inputs[1], y_pred[n]) * weights[n]
                else:
                    curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss += curr_loss
            test_loss += loss.item()

        test_loss /= num_samples
        tb_writer.add_scalar('test_loss', test_loss, epoch + 1)
        print('Test loss: %.4e' % test_loss)

    model.train()


