import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from Reconstruction.model import Generator
from utils import TrainSetLoader_Upsample
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Reconstruction")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=2, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--resume", default="/Artery_Vein/checkpoint/pretrained_discriminator",
                    type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--vgg_loss", default=True, help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=0.9, help='Learning Rate decay')


def l1_loss_with_weight(prediction_results, ground_truth, weight):
    return torch.mean(torch.abs(prediction_results - ground_truth) * weight)


def photo_loss(prediction_results, ground_truth):
    diff_pre = prediction_results[:, 1:] - prediction_results[:, :-1]
    diff_gt = ground_truth[:, 1:] - ground_truth[:, :-1]
    return torch.mean(torch.abs(diff_pre - diff_gt))


def train():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Building model")
    model = Generator()
    # model.load_state_dict(torch.load(
    #         "/Artery_Vein_Upsampling/checkpoint/pretrained_generator/predict_residual/backup/sr_model.pth"))

    print("===> Setting GPU")
    model = model.cuda()
    model = model.to('cuda')

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    print(model)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        print("===> Loading datasets")
        train_set = TrainSetLoader_Upsample('/Train_and_Test/Artery_Vein/upsampling', device)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                          shuffle=True)
        trainor(training_data_loader, optimizer, model, epoch, scheduler)


def trainor(training_data_loader, optimizer, model, epoch, scheduler):
    scheduler.step()
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_epoch = 0
    for iteration, (lr_img, hr_img, weight) in enumerate(training_data_loader):
        residual_pre = model(lr_img[:, :3], lr_img[:, 2:5])
        hr_pre = residual_pre + lr_img[:, 2].unsqueeze(dim=1)

        loss_1 = l1_loss_with_weight(hr_pre, hr_img, weight)
        loss_2 = photo_loss(hr_pre, hr_img)
        loss = loss_1 + 0.1 * loss_2
        loss_epoch += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("===> Epoch[{}]: Loss: {:.5f} loss_avg: {:.5f}".format
              (epoch, loss, loss_epoch / (iteration % 200 + 1)))
        if (iteration + 1) % 200 == 0:
            loss_epoch = 0
            save_checkpoint(model, epoch)
            print("model has benn saved")


def save_checkpoint(model, epoch):
    model_out_path = "/Train_and_Test/Artery_Vein/checkpoint/compare_gene/" \
                     + "model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path, _use_new_zipfile_serialization=False)
    print("Checkpoint saved to {}".format(model_out_path))


train()


