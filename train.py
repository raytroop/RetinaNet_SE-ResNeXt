"""Train the model"""
import sys
import json
import argparse
import logging
import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from models.losses import focal_loss, smooth_l1_loss
from models.fpn import retinanet
from dataGen.data_loader import fetch_trn_loader, fetch_val_loader
import utils

with open('config.json', 'r') as f:
    config = json.load(f)
# torch.device object used throughout this script
device = torch.device("cuda" if config['use_cuda'] else "cpu")

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True, choices=[1, 2, 3, 4, 5], type=int, help="Directory containing the dataset")
    parser.add_argument('--learning_rate', default=1e-5, type=float, help="learning rate of optimizer")
    parser.add_argument('--num_epochs', default=30, type=int, help="total epochs to train")
    parser.add_argument('--frozen_epochs', default=15, type=int, help="the first epoches to fix parameter of backbone")
    parser.add_argument('--save_dir', default='checkpoints', type=str, help="Directory containing params.json")
    parser.add_argument('--checkpoint2load', default=None, type=str, help="checkpoint to load")  # 'best' or 'train'
    parser.add_argument('--optim_restore', default=True, type=bool, help="whether to restore optimizer parameter")
    return parser.parse_args(args)


def train(model, optimizer, loss_fn, dataloader, params, epoch):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    loss_TOTAL = utils.RunningAverage()
    loss_FL = utils.RunningAverage()
    loss_L1 = utils.RunningAverage()

    model.train()
    if epoch < params.frozen_epochs:
        model.train_extractor(False)
    else:
        model.train_extractor(True)

    with tqdm(total=len(dataloader)) as t:
        for i, (img_batch, labels_batch, regression_batch) in enumerate(dataloader):
            img_batch, labels_batch, regression_batch =  img_batch.to(device), labels_batch.to(device), regression_batch.to(device)
            classification_pred, regression_pred, _ = model(img_batch)
            loss_cls = loss_fn['focal'](labels_batch, classification_pred) * config['loss_ratio_FL2L1']
            loss_reg = loss_fn['smooth_l1'](regression_batch, regression_pred)
            loss_all = loss_cls + loss_reg

            loss_cls_detach = loss_cls.detach().item()
            loss_reg_detach = loss_reg.detach().item()
            loss_all_detach = loss_all.detach().item()
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss_all.backward()
            # The norm is computed over all gradients together
            clip_grad_norm_(model.parameters(), 0.5)
            # performs updates using calculated gradients
            optimizer.step()

            # update the average loss
            loss_TOTAL.update(loss_all_detach)
            loss_FL.update(loss_cls_detach)
            loss_L1.update(loss_reg_detach)

            del img_batch, labels_batch, regression_batch

            t.set_postfix(total_loss='{:05.3f}'.format(loss_all_detach), FL_loss='{:05.3f}'.format(
                loss_cls_detach), L1_loss='{:05.3f}'.format(loss_reg_detach))
            t.update()
    logging.info("total_loss:{:05.3f} FL_loss:{:05.3f} L1_loss:{:05.3f}".format(loss_TOTAL(), loss_FL(), loss_L1()))
    del loss_TOTAL, loss_FL, loss_L1


def evaluate(model, loss_fn, val_dataloader, params, epoch):
    # set model to evaluation mode
    model.eval()

    loss_TOTAL = utils.RunningAverage()
    loss_FL = utils.RunningAverage()
    loss_L1 = utils.RunningAverage()

    with torch.no_grad():
        for i, (img_batch, labels_batch, regression_batch) in enumerate(val_dataloader):
            img_batch, labels_batch, regression_batch =  img_batch.to(device), labels_batch.to(device), regression_batch.to(device)
            classification_pred, regression_pred, _ = model(img_batch)
            loss_cls = loss_fn['focal'](labels_batch, classification_pred) * config['loss_ratio_FL2L1']
            loss_reg = loss_fn['smooth_l1'](regression_batch, regression_pred)
            loss_all = loss_cls + loss_reg

            loss_cls_detach = loss_cls.detach().item()
            loss_reg_detach = loss_reg.detach().item()
            loss_all_detach = loss_all.detach().item()

            # update the average loss
            loss_TOTAL.update(loss_all_detach)
            loss_FL.update(loss_cls_detach)
            loss_L1.update(loss_reg_detach)

            del img_batch, labels_batch, regression_batch

    logging.info("total_loss:{:05.3f} FL_loss:{:05.3f} L1_loss:{:05.3f}".format(loss_TOTAL(), loss_FL(), loss_L1()))
    res = loss_TOTAL()
    del loss_TOTAL, loss_FL, loss_L1
    return res

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, params,
                       scheduler=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """
    init_epoch = 0
    best_val_loss = float('inf')

    # reload weights from restore_file if specified
    if params.checkpoint2load is not None:
        checkpoint = utils.load_checkpoint(params.checkpoint2load, model, optimizer if params.optim_restore else None)
        if 'epoch' in checkpoint:
            init_epoch = checkpoint['epoch']
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']

    for epoch in range(init_epoch, params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        if scheduler is not None:
            scheduler.step()
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, params, epoch)

        logging.info("validating ... ")
        # Evaluate for one epoch on validation set
        val_loss = evaluate(model, loss_fn, val_dataloader, params, epoch)

        is_best = val_loss <= best_val_loss
        if is_best:
            best_val_loss = val_loss
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optim_dict': optimizer.state_dict(),
                            'best_val_loss': best_val_loss},
                            is_best=is_best,
                            checkpoint=params.save_dir)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parse_args()

    # Set the random seed for reproducible experiments
    torch.manual_seed(42)
    if config['use_cuda']:
        torch.cuda.manual_seed(42)

    args.save_dir = os.path.join(args.save_dir, config['backbone'], f'fold_{args.fold}')
    # Set the logger
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # save config in file
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        config.update(vars(args))
        json.dump(config, f, indent=4)

    utils.set_logger(os.path.join(args.save_dir, 'train.log'))
    logging.info(' '.join(sys.argv[:]))
    logging.info(args.save_dir)

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    # fetch dataloaders
    train_dl = fetch_trn_loader(args.fold)
    val_dl = fetch_val_loader(args.fold)

    # Define the model and optimizer
    Net = retinanet(config['backbone'])
    model = Net.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # fetch loss function and metrics
    loss_fn = {'focal': focal_loss(alpha=config['focal_alpha']), 'smooth_l1': smooth_l1_loss(sigma=config['l1_sigma'])}

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, args, scheduler=None)
    logging.info('Done')
