import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

import modeling

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Generate .npy file for dataloader
        self.img_process(args) 

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = getattr(modeling, args.model_name)(pretrained=args.pretrained)


        # Define Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
        #                 {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Criterion
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    # 将大图按unit_size的大小，每次stride的移动量进行裁剪。将分好的训练集和验证机以np数组形式存储在save_dir中，
    # 方便下次使用，并减少内存的占用。请将路径修改为自己的。
    def img_process(self, args):
        unit_size = args.base_size
        stride = unit_size # int(unit_size/2)
        save_dir = os.path.join('/data/dingyifeng/pytorch-jingwei-master/npy_process', str(unit_size))
        # npy_process
        if not os.path.exists(save_dir):

            Image.MAX_IMAGE_PIXELS = 100000000000
            # load train image 1
            img = Image.open('/data/dingyifeng/jingwei/jingwei_round1_train_20190619/image_1.png')
            img = np.asarray(img) #(50141, 47161, 4)
            anno_map = Image.open('/data/dingyifeng/jingwei/jingwei_round1_train_20190619/image_1_label.png')
            anno_map = np.asarray(anno_map) #(50141, 47161)

            length, width = img.shape[0], img.shape[1]
            x1, x2, y1, y2 = 0, unit_size ,0 ,unit_size
            Img1 = [] # 保存小图的数组
            Label1 = []  # 保存label的数组
            while(x1 < length):
                #判断横向是否越界
                if x2 > length:
                    x2 , x1 = length , length - unit_size

                while(y1 < width):
                    if y2 > width:
                        y2 , y1  = width , width - unit_size
                    im = img[x1:x2, y1:y2, :]
                    if 255 in im[:,:,-1]:    # 判断裁剪出来的小图中是否存在有像素点
                        Img1.append(im[:,:,0:3])   # 添加小图
                        Label1.append(anno_map[x1:x2, y1:y2])   # 添加label

                    if y2 == width: break

                    y1 += stride
                    y2 += stride

                if x2 == length: break

                y1, y2 = 0, unit_size
                x1 += stride
                x2 += stride
            Img1 = np.array(Img1) #(4123, 448, 448, 3)
            Label1 = np.array(Label1) #(4123, 448, 448)

            # load train image 2
            img = Image.open('/data/dingyifeng/jingwei/jingwei_round1_train_20190619/image_2.png')
            img = np.asarray(img) #(50141, 47161, 4)
            anno_map = Image.open('/data/dingyifeng/jingwei/jingwei_round1_train_20190619/image_2_label.png')
            anno_map = np.asarray(anno_map) #(50141, 47161)

            length, width = img.shape[0], img.shape[1]
            x1, x2, y1, y2 = 0, unit_size ,0 ,unit_size
            Img2 = [] # 保存小图的数组
            Label2 = []  # 保存label的数组
            while(x1 < length):
                #判断横向是否越界
                if x2 > length:
                    x2 , x1 = length , length - unit_size

                while(y1 < width):
                    if y2 > width:
                        y2 , y1  = width , width - unit_size
                    im = img[x1:x2, y1:y2, :]
                    if 255 in im[:,:,-1]:    # 判断裁剪出来的小图中是否存在有像素点
                        Img2.append(im[:,:,0:3])   # 添加小图
                        Label2.append(anno_map[x1:x2, y1:y2])   # 添加label

                    if y2 == width: break

                    y1 += stride
                    y2 += stride

                if x2 == length: break

                y1, y2 = 0, unit_size
                x1 += stride
                x2 += stride
            Img2 = np.array(Img2) #(5072, 448, 448, 3)
            Label2 = np.array(Label2) #(5072, 448, 448)


            Img = np.concatenate((Img1,Img2),axis=0) 
            cat = np.concatenate((Label1,Label2),axis=0) 

            # shuffle
            np.random.seed(1)
            assert(Img.shape[0]==cat.shape[0])
            shuffle_id = np.arange(Img.shape[0])
            np.random.shuffle(shuffle_id)
            Img = Img[shuffle_id]
            cat = cat[shuffle_id]

            os.mkdir(save_dir)
            print("=> generate {}".format(unit_size))
            # split train dataset
            images_train = Img #[:int(Img.shape[0]*0.8)]
            categories_train = cat #[:int(cat.shape[0]*0.8)]
            assert (len(images_train) == len(categories_train))
            np.save(os.path.join(save_dir, 'train_img.npy'), images_train)
            np.save(os.path.join(save_dir, 'train_label.npy'), categories_train)
            # split val dataset
            images_val = Img[int(Img.shape[0]*0.8):]
            categories_val = cat[int(cat.shape[0]*0.8):]
            assert (len(images_val) == len(categories_val))
            np.save(os.path.join(save_dir, 'val_img.npy'), images_val)
            np.save(os.path.join(save_dir, 'val_label.npy'), categories_val)

            print("=> img_process finished!")
        else:
            print("{} file already exists!".format(unit_size))
        for x in locals().keys():
            del locals()[x]
        # 释放内存
        import gc
        gc.collect()

                        

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch UNet Training")
    parser.add_argument("--model-name", type=str, default="UNetResNet34",
                        choices=['UNetResNet34', 'UNetSimple'],
                        help='name of model')
    parser.add_argument("--pretrained", help='use pretrained model',
                        action='store_true', default=False)
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='jingwei',
                        choices=['pascal', 'coco', 'cityscapes', 'jingwei'],
                        help='dataset name (default: jingwei)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'jingwei': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'jingwei': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
