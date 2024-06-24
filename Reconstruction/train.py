import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from losses import reconstruction_loss
import torch.optim as optim
from torch.utils.data import DataLoader
from model import I2SR
from utils import TrainSetLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=2, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=10, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--vgg_loss", default=True, help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=0.8
                    , help='Learning Rate decay')


def train():
    opt = parser.parse_args()
    cuda = opt.cuda
    print("=> use gpu id: '{}'".format(opt.gpus))
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    model = I2SR(nc=32).cuda()
    model.load_state_dict(
        torch.load("/data/models/reconstruction/I2SR_1.pth"))

    # model = RFDN(in_nc=5, out_nc=5, nf=64, upscale=1).cuda()
    # model.load_state_dict(
    #         torch.load("/data/models/reconstruction/RFDN_1.pth"))

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    print(model)

    print("===> Setting Optimizer")
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        data_set = TrainSetLoader(device)
        data_loader = DataLoader(dataset=data_set, num_workers=opt.threads,
                                 batch_size=opt.batchSize, shuffle=True)
        trainor(data_loader, optimizer, model, epoch)
        scheduler.step()
        # seg_scheduler.step()


def trainor(data_loader, optimizer, model, epoch):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_epoch = 0
    for iteration, (raw, gt) in enumerate(data_loader):
        # print(raw.shape, gt.shape)

        # raw = raw.to(device, non_blocking=True)
        # gt = gt.to(device, non_blocking=True)
        pre = model(raw)
        # print(pre.shape)
        # for i in range(pre.shape[1]):
        #     pre[:, i, :, :] += raw[:, 2, :, :]

        # view_1 = raw.cpu().detach().numpy()
        # view_2 = pre.cpu().detach().numpy()
        # view_3 = gt.cpu().detach().numpy()
        # plot_parallel(
        #     a=view_1[0, 2, :, :],
        #     b=view_2[0, 0, :, :],
        #     c=view_3[0, 0, :, :]
        # )
        # print(compare_img(view_1[0, 2, :, :], view_3[0, 0, :, :]))
        # print(compare_img(view_2[0, 0, :, :], view_3[0, 0, :, :]))
        loss = reconstruction_loss(pre, gt, lamb=0.3, dim="2d")
        # for sub_index in range(1, 5):
        #     sub_mask = F.interpolate(mask, scale_factor=2 ** (-1 * sub_index))
        #     # print(sub_mask.shape, pre[sub_index].shape)
        #     loss += sigmoid_focal_loss(pre[sub_index], sub_mask).mean() * 0.1
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # dice = dice_loss(torch.sigmoid(pre[0]), mask)
        loss_epoch += loss

        print("===> Epoch[{}]: loss: {:.5f}  avg_loss: {:.5f}".format
              (epoch, loss, loss_epoch / (iteration % 100 + 1)))

        if (iteration + 1) % 100 == 0:
            loss_epoch = 0
            save_checkpoint(model, epoch, "/data/models/reconstruction")
            # save_checkpoint(seg_model, epoch, "/home/chuy/Artery_Vein_Upsampling/checkpoint/whole/segment/")
            print("model has benn saved")


def save_checkpoint(model, epoch, path):
    model_out_path = os.path.join(path, "re_RFDN_1.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    train()


