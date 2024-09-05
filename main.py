import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
from resnet import resnet18, resnet18_plus
from tqdm import tqdm
from model import *
from utils import *
from fpl import *

from nico_tied import *
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stn',type=bool,default=True,help='run model with stn or not')
    parser.add_argument('--model', type=str, default='resnet18', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='NICO', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox/fpl')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=1000, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=5, help='the mu parameter for fedprox or moon')
    parser.add_argument('--muavg', type=float, default=0.01, help='the mu parameter for fedprox or moon')
    parser.add_argument('--gama', type=float, default=0.1, help='the gama parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.02, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100,
                        help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--experiment', type=str, default='both_update', help='both_update/single_update/shared/fpl/dafkd/fediir')
    args = parser.parse_args()
    return args

def init_nets(net_configs, n_parties, args, device='cuda'):
    nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net = resnet18(args.dataset, kernel_size=3)
        nets[net_i] = net.to(device)
    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, [], model_meta_data, layer_type

def train_net_ori(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))



    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx,(x, target) in enumerate(train_dataloader):

            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()

            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0

def train_net_shared3(net_id, net, net_stn, train_dataloader, test_dataloader, global_model, epochs, lr, args_optimizer,
                     args, device="cpu"):
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    global_model_state_dict = copy.deepcopy(global_model.state_dict())
    net_stn.cuda()
    net_stn.eval()
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()
    cnt = 0
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        #epoch_loss2_collector = []
        epoch_loss3_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            size = x.shape[0]

            optimizer.zero_grad()

            _, x_stn, _ = net_stn(x)

            target = target.long()

            x_new = torch.cat((x_stn, x), dim=0)

            _, feature, out = net(x_new)

            loss1 = criterion(out[size:, :], target)

            #loss2 = kl_divergence(feature[:size, :], feature[size:, :], 'cuda')

            loss3 = criterion(out[:size, :], target)

            loss = loss1 + loss3

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            #epoch_loss2_collector.append(loss2.item())
            epoch_loss3_collector.append(loss3.item())
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        #epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss3: %f' % (
        epoch, epoch_loss, epoch_loss1,  epoch_loss3))
    global_model.load_state_dict(global_model_state_dict)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0

def train_net_shared2(net_id, net, net_stn, train_dataloader, test_dataloader, global_model, epochs, lr, args_optimizer,
                     args, device="cpu"):

    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    global_model_state_dict = copy.deepcopy(global_model.state_dict())
    net_stn.cuda()
    net_stn.eval()
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()
    cnt = 0
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        epoch_loss3_collector =[]
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            size = x.shape[0]
            optimizer.zero_grad()
            _, x_stn, _ = net_stn(x)
            target = target.long()
            x_new = torch.cat((x_stn, x), dim=0)
            _, feature, out = net(x_new)
            loss1 = criterion(out[size:, :], target)
            loss2 = kl_divergence(feature[:size, :], feature[size:, :], 'cuda')
            loss3 = criterion(out[:size, :], target)
            loss = loss1 + args.muavg * loss2 + loss3
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())
            epoch_loss3_collector.append(loss3.item())
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        epoch_loss3 = sum(epoch_loss3_collector) /len(epoch_loss3_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f Loss3: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2,epoch_loss3))
    global_model.load_state_dict(global_model_state_dict)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return 0, 0

#实验一:共享本地模型进行更新
def train_net_shared(net_id, net, net_stn, train_dataloader, test_dataloader, global_model, epochs, lr, args_optimizer,
                     args, device="cpu"):
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    global_model_state_dict = copy.deepcopy(global_model.state_dict())
    net_stn.cuda()
    net_stn.eval()

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []

        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            size = x.shape[0]

            optimizer.zero_grad()

            _, x_stn, _ = net_stn(x)

            target = target.long()

            x_new = torch.cat((x_stn, x), dim=0)

            _, feature, out = net(x_new)

            loss1 = criterion(out[size:, :], target)

            loss2 = kl_divergence(feature[:size, :], feature[size:, :], 'cuda')

            loss = loss1 + args.muavg * loss2

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)

        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f ' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))
    global_model.load_state_dict(global_model_state_dict)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return 0, 0

#实验二:在每个客户端内先提前预训练，然后只在上面支路更新模型
def train_net_single(net_id, net, net_stn, train_dataloader, test_dataloader, global_model, epochs, lr, args_optimizer,
                     args, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    global_model_state_dict = copy.deepcopy(global_model.state_dict())
    for param in net_stn.parameters():
        param.requires_grad = True

    net_stn.cuda()
    net_stn.classification.load_state_dict(global_model.state_dict())
    net_stn.train()
    optimizer_stn = optim.SGD(net_stn.parameters(), lr=0.1)
    cnt2 = 0
    for subepoch in range(10):
        subepoch_loss_collector = []

        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to('cuda'), target.to('cuda')
            target = target.long()
            optimizer_stn.zero_grad()
            output, _, _ = net_stn(data)
            loss_stn = F.nll_loss(output, target)
            loss_stn.backward()
            optimizer_stn.step()
            cnt2 += 1
            subepoch_loss_collector.append(loss_stn.item())

        subepoch_loss = sum(subepoch_loss_collector) / len(subepoch_loss_collector)
        logger.info('Pretrain_SUBEpoch: %d SUB_Loss: %f ' % (subepoch, subepoch_loss))

    net_stn.eval()

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []

        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            _, x_stn, feature = net_stn(x)
            target = target.long()

            _, feature1, out = net(x)

            loss1 = criterion(out, target)

            loss2 = kl_divergence(feature1, feature, 'cuda')

            loss = loss1 + args.muavg * loss2

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)

        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f ' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))
    global_model.load_state_dict(global_model_state_dict)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return 0, 0

#实验三:上下通路都更新
def train_net(net_id, net, net_stn, train_dataloader, test_dataloader, global_model, epochs, lr, args_optimizer, args,round,
              device="cpu"):
    #net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    global_model_state_dict = copy.deepcopy(global_model.state_dict())
    for param in net_stn.parameters():
        param.requires_grad = True
    #global_model.train()
    #net_stn.cuda()

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, global_model.parameters()), lr=lr, momentum=0.9,
                               weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()
    cnt = 0
    if round!=0:
     for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        #epoch_loss3_collector = []
        for batch_idx, (x,x_2, target) in enumerate(train_dataloader):
            x,x_2, target = x.cuda(), x_2.cuda(),target.cuda()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            # x_stn = net_stn.stn(x)
            #_, x_stn, _ = net_stn(x)
            target = target.long()

            _, feature1, out = net(x)
            loss1 = criterion(out, target)


            _, feature2, out2 = global_model(x_2)
            loss2 = kl_divergence(feature1, feature2, 'cuda')
            loss = loss1 + args.muavg * loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        #epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f ' % (
        epoch, epoch_loss, epoch_loss1, epoch_loss2))
    else:
     for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        #epoch_loss2_collector = []
        # epoch_loss3_collector = []
        for batch_idx, (x, x_2, target) in enumerate(train_dataloader):
            x, x_2, target = x.cuda(), x_2.cuda(), target.cuda()
            optimizer.zero_grad()
            optimizer2.zero_grad()
            # x_stn = net_stn.stn(x)
            # _, x_stn, _ = net_stn(x)
            target = target.long()

            _, feature1, out = net(x)
            loss1 = criterion(out, target)

            #_, feature2, out2 = global_model(x_2)
            #loss2 = kl_divergence(feature1, feature2, 'cuda')
            loss = loss1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            #epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        #epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        # epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f ' % (
            epoch, epoch_loss, epoch_loss1))


    global_model.load_state_dict(global_model_state_dict)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return 0, 0


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu"):
    # global_net.to(device)
    net = nn.DataParallel(net)
    net.cuda()
    # else:
    #     net.to(device)
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()
    cnt = 0
    global_weight_collector = list(global_net.cuda().parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()

            target = target.long()

            _, _, out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()
            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0,0


def train_net_fedcon_shared2(net_id, net, net_stn, global_net, previous_nets, train_dataloader, test_dataloader, epochs,
                            lr,
                            args_optimizer, mu, temperature, args,
                            round, device="cpu"):
    # net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    global_model_state_dict = copy.deepcopy(global_model.state_dict())
    net_stn.cuda()
    net_stn.eval()

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    global_net.cuda()

    for previous_net in previous_nets:
        previous_net.cuda()
    # global_w = global_net.state_dict()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    global_net_state_dict = copy.deepcopy(global_net.state_dict())
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        epoch_loss3_collector = []
        # epoch_loss4_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            # altered_section
            optimizer.zero_grad()
            size = x.shape[0]
            x_stn = net_stn.stn(x)
            # _, x_stn, _ = net_stn(x)
            target = target.long()
            x_new = torch.cat((x, x_stn), dim=0)
            h, f, out = net(x_new)
            # h1,f1,out1 = net(x)
            # h_g,f_g,out_g = global_net(x_stn)

            posi = cos(f[:size, :], f[size:, :])
            logits = posi.reshape(-1, 1)

            for previous_net in previous_nets:
                previous_net.cuda()

                _, f3, _ = previous_net(x)

                nega = cos(f[:size, :], f3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0).cuda().long())

            loss2 = mu * criterion(logits, labels)

            loss1 = criterion(out[:size, :], target)

            loss3=criterion(out[size:,:],target)

            loss = loss1 + loss2 +loss3

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())
            epoch_loss3_collector.append(loss3.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)

        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f Loss3: %f ' % (
            epoch, epoch_loss, epoch_loss1, epoch_loss2,epoch_loss3))

    global_net.load_state_dict(global_net_state_dict)
    for previous_net in previous_nets:
        previous_net.to('cpu')

    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0

def train_net_fedcon_shared(net_id, net, net_stn, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                     args_optimizer, mu, temperature, args,
                     round, device="cpu"):
    # net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    global_model_state_dict = copy.deepcopy(global_model.state_dict())
    net_stn.cuda()
    net_stn.eval()

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)


    criterion = nn.CrossEntropyLoss().cuda()

    global_net.cuda()

    for previous_net in previous_nets:
        previous_net.cuda()
    # global_w = global_net.state_dict()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    global_net_state_dict = copy.deepcopy(global_net.state_dict())
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        #epoch_loss3_collector = []
        # epoch_loss4_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            # altered_section
            optimizer.zero_grad()
            size = x.shape[0]
            #x_stn = net_stn.stn(x)
            _, x_stn, _ = net_stn(x)
            target = target.long()
            x_new = torch.cat((x, x_stn), dim=0)
            h,f,out=net(x_new)
            #h1,f1,out1 = net(x)
            #h_g,f_g,out_g = global_net(x_stn)



            posi = cos(f[:size,:], f[size:,:])
            logits = posi.reshape(-1, 1)


            for previous_net in previous_nets:
                previous_net.cuda()

                _,f3,_= previous_net(x)


                nega = cos(f[:size,:], f3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                previous_net.to('cpu')

            logits/=temperature

            labels=torch.zeros(x.size(0)).cuda().long()

            loss2=mu*criterion(logits,labels)

            loss1=criterion(out[:size,:],target)

            #loss3=criterion(out[size:,:],target)

            loss=loss1+loss2

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()



            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())
            #epoch_loss3_collector.append(loss3.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        #epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)

        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f ' % (
        epoch, epoch_loss, epoch_loss1, epoch_loss2))

    global_net.load_state_dict(global_net_state_dict)
    for previous_net in previous_nets:
        previous_net.to('cpu')

    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0

#实验一:只有一条通路更新
def train_net_fediir(net_id, net, downloads, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                     args_optimizer, mu, temperature, args,
                     round, device="cpu"):
    global_net_state_dict = copy.deepcopy(global_net.state_dict())
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))
    downloads.cuda()
    global_net.cuda()
    for param in downloads.parameters():
        param.requires_grad = True
    downloads.cuda()
    downloads.classification.load_state_dict(global_model.state_dict())
    downloads.train()
    optimizer_stn = optim.SGD(downloads.parameters(), lr=0.1)
    cnt2 = 0
    for subepoch in range(10):
        subepoch_loss_collector = []
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to('cuda'), target.to('cuda')
            target = target.long()
            optimizer_stn.zero_grad()
            output, _, _ = downloads(data)
            loss_stn = F.nll_loss(output, target)
            loss_stn.backward()
            optimizer_stn.step()
            cnt2 += 1
            subepoch_loss_collector.append(loss_stn.item())
        subepoch_loss = sum(subepoch_loss_collector) / len(subepoch_loss_collector)
        logger.info('Pretrain_SUBEpoch: %d SUB_Loss: %f ' % (subepoch, subepoch_loss))
    downloads.eval()
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()
    global_net.cuda()
    for previous_net in previous_nets:
        previous_net.cuda()
    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            # altered_section
            optimizer.zero_grad()
            _, x_stn,feature= downloads(x)
            target = target.long()
            h1,f1,out1 = net(x)
            posi = cos(f1, feature)
            logits = posi.reshape(-1, 1)
            for previous_net in previous_nets:
                previous_net.cuda()
                _,f3,_= previous_net(x)
                nega = cos(f1, f3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                previous_net.to('cpu')
            logits/=temperature
            labels=torch.zeros(x.size(0).cuda().long())
            loss2=mu*criterion(logits,labels)
            loss1=criterion(out1,target)
            loss=loss1+loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)


        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f ' % (
        epoch, epoch_loss, epoch_loss1, epoch_loss2))

    global_net.load_state_dict(global_net_state_dict)
    for previous_net in previous_nets:
        previous_net.to('cpu')

    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0

def train_net_fedcon_ori(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args,
                      round, device="cpu"):
    #net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    global_net.cuda()

    for previous_net in previous_nets:
        previous_net.cuda()
    # global_w = global_net.state_dict()

    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()

            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1,1)

            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)


            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    for previous_net in previous_nets:
        previous_net.to('cpu')

    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0,0

#两条通路都更新
def train_net_fedcon(net_id, net, net_stn, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                     args_optimizer, mu, temperature, args,
                     round, device="cpu"):
    # net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    net_stn.cuda()
    global_net.cuda()
    for param in net_stn.parameters():
        param.requires_grad = True

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, global_net.parameters()), lr=lr, momentum=0.9,
                                  weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    global_net.cuda()

    for previous_net in previous_nets:
        previous_net.cuda()
    # global_w = global_net.state_dict()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    global_net_state_dict = copy.deepcopy(global_net.state_dict())
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        epoch_loss3_collector = []
        # epoch_loss4_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            # altered_section
            optimizer.zero_grad()
            optimizer2.zero_grad()

            # x_stn = net_stn.stn(x)
            _, x_stn, _ = net_stn(x)
            target = target.long()

            h1,f1,out1 = net(x)
            h_g,f_g,out_g = global_net(x_stn)



            posi = cos(f1, f_g)
            logits = posi.reshape(-1, 1)


            for previous_net in previous_nets:
                previous_net.cuda()

                _,f3,_= previous_net(x)


                nega = cos(f1, f3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                previous_net.to('cpu')

            logits/=temperature
            labels=torch.zeros(x.size(0).cuda().long())

            loss2=mu*criterion(logits,labels)

            loss1=criterion(out1,target)

            loss3=criterion(out_g,target)

            loss=loss1+loss2

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

            loss3.backward()
            optimizer2.step()
            optimizer2.zero_grad()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())
            epoch_loss3_collector.append(loss3.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        epoch_loss3 = sum(epoch_loss3_collector) / len(epoch_loss3_collector)

        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f Loss3: %f' % (
        epoch, epoch_loss, epoch_loss1, epoch_loss2, epoch_loss3))

    global_net.load_state_dict(global_net_state_dict)
    for previous_net in previous_nets:
        previous_net.to('cpu')

    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0

def train_net_dafkd(net_id, net, net_stn, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                     args_optimizer, mu, temperature, args,
                     round, device="cpu"):
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    net_stn.cuda()
    global_net.cuda()
    for param in net_stn.parameters():
        param.requires_grad = True

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, global_net.parameters()), lr=lr, momentum=0.9,
                                  weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    global_net.cuda()

    for previous_net in previous_nets:
        previous_net.cuda()
    # global_w = global_net.state_dict()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001
    from model import DaFKD
    DaFKD=DaFKD(train_dataloader,device,args,None)
    global_net_state_dict = copy.deepcopy(global_net.state_dict())
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        # epoch_loss4_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            # altered_section
            optimizer.zero_grad()
            optimizer2.zero_grad()
            _, x_stn, _ = net_stn(x)
            target = target.long()
            h1,f1,out1 = net(x)
            h_g,f_g,out_g = global_net(x_stn)
            posi = cos(f1, f_g)
            logits = posi.reshape(-1, 1)
            for previous_net in previous_nets:
                previous_net.cuda()
                _,f3,_= previous_net(x)
                nega = cos(f1, f3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                previous_net.to('cpu')
            logits/=temperature
            labels=torch.zeros(x.size(0).cuda().long())
            loss2=mu*criterion(logits,labels)
            loss1=criterion(out1,target)
            loss=loss1+loss2
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f ' % (
        epoch, epoch_loss, epoch_loss1))
    global_net.load_state_dict(global_net_state_dict)
    for previous_net in previous_nets:
        previous_net.to('cpu')

    net.to('cpu')
    logger.info(' ** Training complete **')
    return 0, 0

def train_fpl_net(net_id, net, train_dataloader, epochs, lr,args_optimizer, temperature,agg_protos_class):
    net = net.to('cuda')
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    criterion.to('cuda')

    if len(agg_protos_class) != 0:
        all_global_protos_keys = np.array(list(agg_protos_class.keys()))
        all_f = []
        mean_f = []
        for protos_key in all_global_protos_keys:
            temp_f = agg_protos_class[protos_key]
            temp_f = torch.cat(temp_f, dim=0).to('cuda')
            all_f.append(temp_f.cpu())
            mean_f.append(torch.mean(temp_f, dim=0).cpu())
        # cluster proto
        all_f = [item.detach() for item in all_f]
        # unbiased proto
        mean_f = [item.detach() for item in mean_f]

    agg_protos_label = {}
    for epoch in range(epochs):

        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            images = images.to('cuda')
            labels = labels.to('cuda')

            #f = net.features(images)
            #outputs = net.classifier(f)
            _,f,outputs=net(images)

            lossCE = criterion(outputs, labels.long())

            if len(agg_protos_class) == 0:
                loss_InfoNCE = 0 * lossCE
            else:
                i = 0
                loss_InfoNCE = None

                for label in labels:
                    if label.item() in agg_protos_class.keys():
                        f_now = f[i].unsqueeze(0)
                        loss_instance = hierarchical_info_loss(f_now, label, all_f, mean_f, all_global_protos_keys,temperature)

                        if loss_InfoNCE is None:
                            loss_InfoNCE = loss_instance
                        else:
                            loss_InfoNCE += loss_instance
                    i += 1
                loss_InfoNCE = loss_InfoNCE / i
            loss_InfoNCE = loss_InfoNCE

            loss = lossCE + loss_InfoNCE
            loss.backward()

            optimizer.step()

            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(lossCE.item())
            epoch_loss2_collector.append(loss_InfoNCE.item())

            if epoch == epochs - 1:
                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        agg_protos_label[labels[i].item()].append(f[i, :])
                    else:
                        agg_protos_label[labels[i].item()] = [f[i, :]]
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f LossCE: %f Loss_InfoNCE: %f ' % (
            epoch, epoch_loss, epoch_loss1, epoch_loss2))
    agg_protos = agg_func(agg_protos_label)
    #self.local_protos[index] = agg_protos

    return 0,0,agg_protos


def local_train_net(nets, nets_stn,args, net_dataidx_map, train_dl=None, test_dl=None, global_model=None,
                    prev_model_pool=None,
                    server_c=None, clients_c=None, round=None,train_data=None,test_data=None, device="cpu",agg_protos_class=None):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()
        for param in global_model.parameters():
            param.requires_grad = True
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    if args.alg=='fpl':
        local_protos={}
    for net_id, net in nets.items():
        if nets_stn:
            net_stn = nets_stn[net_id]
        #print(net_dataidx_map[0])
        if args.dataset!='color_mnist'and args.dataset!='NICO':
            dataidxs = net_dataidx_map[net_id]

        #logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        if args.dataset=='color_mnist':
              train_dl_local,_=get_color_mnist_dataloader(net_id,args.beta)
        elif args.dataset=='NICO':
              train_dl_local,_=get_NICO_dataloader_train(net_id)

        else:
              train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
              train_dl_global, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        if args.alg == 'fedavg':
            if args.experiment == 'both_update':
                trainacc, testacc = train_net(net_id, net, [], train_dl_local, test_dl, global_model, n_epoch,
                                              args.lr, args.optimizer, args,
                                              device=device,round=round)
            elif args.experiment == 'single_update':
                trainacc, testacc = train_net_single(net_id, net, net_stn, train_dl_local, test_dl, global_model,
                                                     n_epoch,
                                                     args.lr, args.optimizer, args,
                                                     device=device)
            elif args.experiment=='shared':
                trainacc, testacc = train_net_shared(net_id, net, net_stn, train_dl_local, test_dl, global_model,
                                                     n_epoch,
                                                     args.lr, args.optimizer, args,
                                                     device=device)
            elif args.experiment=='ori':
                trainacc,testacc=train_net_ori(net_id,net,train_dl_local,test_dl,n_epoch,
                                               args.lr,args.optimizer,args,device=device)
            elif args.experiment=='shared2':
                trainacc, testacc = train_net_shared2(net_id, net, net_stn, train_dl_local, test_dl, global_model,
                                                     n_epoch,
                                                     args.lr, args.optimizer, args,
                                                     device=device)
            elif args.experiment=='shared3':
                trainacc, testacc = train_net_shared3(net_id, net, net_stn, train_dl_local, test_dl, global_model,
                                                     n_epoch,
                                                     args.lr, args.optimizer, args,
                                                     device=device)
        elif args.alg == 'fedprox':
            trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args.mu, args, device=device)
        elif args.alg == 'fpl':
            trainacc ,testacc,agg_proto_client =train_fpl_net(net_id, net, train_dl_local, n_epoch, args.lr,args.optimizer, args.temperature,agg_protos_class)
            local_protos[net_id]=agg_proto_client
        elif args.alg == 'dafkd':
            trainacc, testacc = train_net_shared3(net_id, net, net_stn, train_dl_local, test_dl, global_model,
                                                     n_epoch,
                                                     args.lr, args.optimizer, args,
                                                     device=device)
        elif args.alg == 'moon':
            prev_models = []
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            if args.experiment=='both_update':
                trainacc, testacc = train_net_fedcon(net_id, net, net_stn, global_model, prev_models, train_dl_local, test_dl,
                                                 n_epoch, args.lr,
                                                 args.optimizer, args.mu, args.temperature, args, round, device=device)
            elif args.experiment=='fediir':
                trainacc, testacc = train_net_fediir(net_id, net, net_stn, global_model, prev_models, train_dl_local,
                                                     test_dl,
                                                     n_epoch, args.lr,
                                                     args.optimizer, args.mu, args.temperature, args, round,
                                                     device=device)
            elif args.experiment=='shared':
                trainacc, testacc = train_net_fedcon_shared(net_id, net, net_stn, global_model, prev_models,
                                                            train_dl_local, test_dl,
                                                            n_epoch, args.lr,
                                                            args.optimizer, args.mu, args.temperature, args, round,
                                                            device=device)
            elif args.experiment=='shared2':
                trainacc, testacc = train_net_fedcon_shared2(net_id, net, net_stn, global_model, prev_models,
                                                            train_dl_local, test_dl,
                                                            n_epoch, args.lr,
                                                            args.optimizer, args.mu, args.temperature, args, round,
                                                            device=device)

            elif args.experiment=='ori':
                trainacc, testacc = train_net_fedcon_ori(net_id, net, global_model, prev_models,
                                                             train_dl_local, test_dl,
                                                             n_epoch, args.lr,
                                                             args.optimizer, args.mu, args.temperature, args, round,
                                                             device=device)
        elif args.alg == 'local_training':
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args,
                                          device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        for param in global_model.parameters():
            param.requires_grad = False
        global_model.to('cpu')

    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    if args.alg=='fpl':
        return local_protos

    return nets

if __name__ == '__main__':

    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    if args.dataset!='color_mnist' and args.dataset!='NICO':
         X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
                  args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    #party_list_rounds中每个元素内包含的是每一轮需要进行参与的clients的索引
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)
    #n_classes = len(np.unique(y_train))
    if args.dataset=='color_mnist':
       train_dl_global, test_dl= get_color_mnist_dataloader(0,beta=args.beta)
    elif args.dataset=='NICO':
       train_dl_global=get_NICO_dataloader_train(0)
       test_dl=get_NICO_dataloader_test()
    else:
       train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)
    #print("len train_dl_global:", len(train_ds_global))
    train_dl = None
    #data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets,nets_stn, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models,_, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    if args.alg=='dafkd' or args.alg=='fediir':
       agg_protos_class={}
       for round in range(n_comm_rounds):

            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_protos=local_train_net(nets_this_round,nets_stn=None, args=args, net_dataidx_map=net_dataidx_map, global_model=global_model,
                                train_dl=train_dl, test_dl=test_dl, device=device,agg_protos_class=agg_protos_class)

            agg_protos_class=proto_aggregation(args.n_parties,local_protos)
            #net_dataidx_map=np.load('net_dataidx_map_fmnist_beta{}.npy'.format(args.beta),allow_pickle=True).reshape(-1)[0]

            if args.dataset=='NICO':
                total_data_points = 10633
                fed_avg_freqs = [1714 / 10633, 1448 / 10633, 1463 / 10633, 1503 / 10633, 1603 / 10633, 1526 / 10633,
                             1376 / 10633]
            else:
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)

            # logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedavg_stn/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(),
                       args.modeldir + 'fedavg_stn/' + 'globalmodel' + args.log_file_name + '.pth')
            torch.save(nets[0].state_dict(),
                       args.modeldir + 'fedavg_stn/' + 'localmodel0' + args.log_file_name + '.pth')
    if args.alg=='fpl':
       agg_protos_class={}
       for round in range(n_comm_rounds):

            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_protos=local_train_net(nets_this_round,nets_stn=None, args=args, net_dataidx_map=None, global_model=global_model,
                                train_dl=train_dl, test_dl=test_dl, device=device,agg_protos_class=agg_protos_class)

            agg_protos_class=proto_aggregation(args.n_parties,local_protos)
            #net_dataidx_map=np.load('net_dataidx_map_fmnist_beta{}.npy'.format(args.beta),allow_pickle=True).reshape(-1)[0]

            if args.dataset=='NICO':
                total_data_points = 10633
                fed_avg_freqs = [1714 / 10633, 1448 / 10633, 1463 / 10633, 1503 / 10633, 1603 / 10633, 1526 / 10633,
                             1376 / 10633]
            else:
                total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)

            # logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedavg_stn/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(),
                       args.modeldir + 'fedavg_stn/' + 'globalmodel' + args.log_file_name + '.pth')
            torch.save(nets[0].state_dict(),
                       args.modeldir + 'fedavg_stn/' + 'localmodel0' + args.log_file_name + '.pth')
    elif args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_' + 'net' + str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            #nets_stn_this_round={k :nets_stn[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round,nets_stn=None, args=args, net_dataidx_map=None, train_dl=train_dl, test_dl=test_dl,
                            global_model=global_model, prev_model_pool=old_nets_pool, round=round, device=device)

            #total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            #fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            total_data_points = 10633

            fed_avg_freqs = [1714 / 10633, 1448 / 10633, 1463 / 10633, 1503 / 10633, 1603 / 10633, 1526 / 10633,
                             1376 / 10633]
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            # summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)
            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)

            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i + 1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir + 'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(),
                           args.modeldir + 'fedcon/global_model_' + args.log_file_name + '.pth')
                torch.save(nets[0].state_dict(), args.modeldir + 'fedcon/localmodel0' + args.log_file_name + '.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool' + str(nets_id) + '_' + 'net' + str(net_id): net.state_dict() for net_id, net in
                                old_nets.items()},
                               args.modeldir + 'fedcon/prev_model_pool_' + args.log_file_name + '.pth')
    elif args.alg == 'fedavg':
        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            #nets_stn_round = {k: nets_stn[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            #net_dataidx_map=np.load('net_dataidx_map_fmnist_beta{}.npy'.format(args.beta),allow_pickle=True).reshape(-1)[0]
            local_train_net(nets_this_round, nets_stn=[], args=args, net_dataidx_map=None, global_model=global_model,
                            train_dl=train_dl, test_dl=test_dl, device=device,round=round)

            #total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            total_data_points=10633
            #fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            fed_avg_freqs=[1714/10633,1448/10633,1463/10633,1503/10633,1603/10633,1526/10633,1376/10633]
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)

            # logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedavg_stn/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(),
                       args.modeldir + 'fedavg_stn/' + 'globalmodel' + args.log_file_name + '.pth')
            torch.save(nets[0].state_dict(),
                       args.modeldir + 'fedavg_stn/' + 'localmodel0' + args.log_file_name + '.pth')
    elif args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args=args, net_dataidx_map=None, train_dl=train_dl, test_dl=test_dl,
                            global_model=global_model, device=device)
            global_model.to('cpu')

            # update global model
            #total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            #fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            total_data_points = 10633
            # fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
            fed_avg_freqs = [1714 / 10633, 1448 / 10633, 1463 / 10633, 1503 / 10633, 1603 / 10633, 1526 / 10633,
                             1376 / 10633]
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)
            mkdirs(args.modeldir + 'fedprox/')
            global_model.to('cpu')
            torch.save(global_model.state_dict(), args.modeldir + 'fedprox/' + args.log_file_name + '.pth')

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(),
                       args.modeldir + 'localmodel/' + 'model' + str(net_id) + args.log_file_name + '.pth')

    elif args.alg == 'all_in':
        nets, _, _ = init_nets(args.net_config, 1, args, device='cpu')
        # nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl, args.epochs, args.lr,
                                      args.optimizer, args, device=device)
        logger.info("All in test acc: %f" % testacc)
        mkdirs(args.modeldir + 'all_in/')
        torch.save(nets[0].state_dict(), args.modeldir + 'all_in/' + args.log_file_name + '.pth')