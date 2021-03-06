import argparse
import os
from statistics import mode
import time
import warnings

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

from dataset import get_dataloaders
from utils import (Logger, get_model, mixup_criterion, mixup_data, random_seed, save_checkpoint, smooth_one_hot,
                   cross_entropy, batch_weight_triplet_loss)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='USTC Computer Vision Final Project')
parser.add_argument('--arch', default="ResNet18", type=str)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--scheduler', default="reduce", type=str, help='[reduce, cos]')
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--label_smooth', default=True, type=eval)
parser.add_argument('--batch_weight', default=True, type=eval)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--label_smooth_value', default=0.1, type=float)
parser.add_argument('--mixup', default=True, type=eval)
parser.add_argument('--mixup_alpha', default=1.0, type=float)
parser.add_argument('--Ncrop', default=False, type=eval)
parser.add_argument('--data_path', default='datasets/fer2013/fer2013.csv', type=str)
parser.add_argument('--results', default='./results', type=str)
parser.add_argument('--save_freq', default=10, type=int)
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--name', default='official', type=str)

best_acc = 0


def main():
    global best_acc

    args = parser.parse_args()
    if random_seed is not None:
        random_seed(args.seed)

    args_path = str(args.arch) + '_epoch' + str(args.epochs) + '_bs' + str(args.batch_size) + '_lr' + str(
        args.lr) + '_momentum' + str(args.momentum) + '_wd' + str(args.weight_decay) + '_seed' + str(
        args.seed) + '_smooth' + str(args.label_smooth) + '_mixup' + str(args.mixup) + '_scheduler' + str(
        args.scheduler) + '_' + str(args.name)

    checkpoint_path = os.path.join(
        args.results, args.name, args_path, 'checkpoints')

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    logger = Logger(os.path.join(args.results,
                                 args.name, args_path, 'output.log'))

    logger.info(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(device)

    logger.info('Load dataset ...')

    train_loader, val_loader, test_loader = get_dataloaders(
        path=args.data_path,
        bs=args.batch_size, Ncrop=args.Ncrop, augment=True)

    logger.info('Start load model %s ...', args.arch)
    model = get_model(args.arch)
    # print(model)

    model = model.to(device)
    # amp
    scaler = GradScaler()

    if args.label_smooth:
        loss_fn = cross_entropy
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.scheduler == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.75, patience=5, verbose=True)

    if args.resume > 0:
        logger.info('Resume from epoch %d', (args.resume))
        state_dict = torch.load(os.path.join(
            checkpoint_path, 'checkpoint_' + str(args.resume) + '.tar'))
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['opt_state_dict'])
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    logger.info('Start traning.')
    logger.info(
        "Epoch \t Time \t Train Loss \t Train ACC \t Val Loss \t Val ACC")
    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        train_loss, train_acc = train(
            model, train_loader, loss_fn, optimizer, epoch, device, scaler, args)
        val_loss, val_acc = evaluate(model, val_loader, device, args)

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'reduce':
            scheduler.step(val_acc)

        epoch_time = time.time() - start_t
        logger.info("%d\t %.4f \t %.4f \t %.4f \t %.4f \t %.4f", epoch, epoch_time, train_loss, train_acc, val_loss,
                    val_acc)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            'best_acc': best_acc,
        }, epoch, is_best, save_path=checkpoint_path, save_freq=args.save_freq)

    logger.info("Best val ACC %.4f", best_acc)


def train(model, train_loader, loss_fn, optimizer, epoch, device, scaler, args):
    model.train()
    count = 0
    correct = 0
    train_loss = 0
    for i, data in enumerate(train_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        with autocast():
            if args.Ncrop:
                bs, ncrops, c, h, w = images.shape
                images = images.view(-1, c, h, w)
                labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)

            if args.mixup:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, args.mixup_alpha)
                images, labels_a, labels_b = map(
                    Variable, (images, labels_a, labels_b))

            if args.batch_weight:
                outputs, auxi = model(images)
            else:
                outputs = model(images)

            if args.label_smooth:
                if args.mixup:
                    # mixup + label smooth
                    soft_labels_a = smooth_one_hot(
                        labels_a, classes=7, smoothing=args.label_smooth_value)
                    soft_labels_b = smooth_one_hot(
                        labels_b, classes=7, smoothing=args.label_smooth_value)
                    loss = mixup_criterion(
                        loss_fn, outputs, soft_labels_a, soft_labels_b, lam)

                    if args.batch_weight:
                        loss +=args.alpha*batch_weight_triplet_loss(labels, auxi)
                else:
                    # label smoorth
                    soft_labels = smooth_one_hot(
                        labels, classes=7, smoothing=args.label_smooth_value)
                    loss = loss_fn(outputs, soft_labels)
                    if args.batch_weight:
                        loss +=args.alpha*batch_weight_triplet_loss(labels, auxi)
            else:
                if args.mixup:
                    # mixup
                    loss = mixup_criterion(
                        loss_fn, outputs, labels_a, labels_b, lam)

                    if args.batch_weight:
                        loss +=args.alpha*batch_weight_triplet_loss(labels, auxi)
                else:
                    # normal CE
                    loss = loss_fn(outputs, labels)
                    if args.batch_weight:
                        loss +=args.alpha*batch_weight_triplet_loss(labels, auxi)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        count += labels.shape[0]

    return train_loss / count, correct / count


def evaluate(model, val_loader, device, args):
    model.eval()
    count = 0
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            if args.Ncrop:
                # fuse crops and batchsize
                bs, ncrops, c, h, w = images.shape
                images = images.view(-1, c, h, w)

                # forward
                outputs, _ = model(images)

                # combine results across the crops
                outputs = outputs.view(bs, ncrops, -1)
                outputs = torch.sum(outputs, dim=1) / ncrops

            else:
                outputs,_ = model(images)

            loss = nn.CrossEntropyLoss()(outputs, labels)

            val_loss += loss
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            count += labels.shape[0]

        return val_loss / count, correct / count


if __name__ == '__main__':
    main()