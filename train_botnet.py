import argparse
import time
import os
import random
import sys

import torch

from botdet.eval.evaluation import eval_metrics, eval_predictor, PygModelPredictor
from botdet.optim.train_utils import time_since, logging_config
from botdet.optim.earlystop import EarlyStopping
from botdet.models_pyg.gcn_model import GCNModel
from botdet.data.dataset_botnet import BotnetDataset
from botdet.data.dataloader import GraphDataLoader


# ============== some default parameters =============
devid = 0
seed = 0

data_dir = './data/botnet'
data_name = 'chord'
batch_size = 2
in_memory = True
shuffle = False

save_dir = './saved_models'
save_name = 'temp.pt'

in_channels = 1
enc_sizes = [32] * 12
residual_hop = 1
edge_gate = 'none'
num_classes = 2
nodemodel = 'additive'

nheads = [1]  # number of heads in multihead attention, should be a list of length 1 or equal to #layers
att_act = 'lrelu'
att_dropout = 0
att_dir = 'in'  # should be 'out' to work for our featureless botnet graphs
att_combine = 'cat'

deg_norm = 'rw'
aggr = 'add'
dropout = 0.0
bias = True
final = 'proj'    # 'none', 'proj'

learning_rate = 0.005
num_epochs = 50
early_stop = True

# ====================================================


def parse_args():
    parser = argparse.ArgumentParser(description='Training a GCN model.')
    # general setting
    parser.add_argument('--devid', type=int, default=devid, help='device id; -1 for CPU')
    parser.add_argument('--seed', type=int, default=seed, help='random seed')
    parser.add_argument('--logmode', type=str, default='w', help='logging file mode')
    parser.add_argument('--log_interval', type=int, default=96, help='logging interval during training')
    # data loading
    parser.add_argument('--data_dir', type=str, default=data_dir, help='directory to find the dataset')
    parser.add_argument('--data_name', type=str, default=data_name,
                        choices=['chord', 'debru', 'kadem', 'leet', 'c2', 'p2p'], help='name of the botnet topology')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='training batch size')
    # parser.add_argument('--in_memory', action='store_true', help='whether to load all the data into memory')
    parser.add_argument('--in_memory', type=int, default=in_memory, help='whether to load all the data into memory')
    parser.add_argument('--shuffle', type=int, default=shuffle, help='whether to shuffle training data')
    # model
    parser.add_argument('--in_channels', type=int, default=in_channels, help='input node feature size')
    parser.add_argument('--enc_sizes', type=int, nargs='*', default=enc_sizes, help='encoding node feature sizes')
    parser.add_argument('--act', type=str, default='relu', choices=['none', 'lrelu', 'relu', 'elu'],
                        help='non-linear activation function after adding residual')
    parser.add_argument('--layer_act', type=str, default='none', choices=['none', 'lrelu', 'relu', 'elu'],
                        help='non-linear activation function for each layer before residual')
    parser.add_argument('--residual_hop', type=int, default=residual_hop, help='residual per # layers')
    parser.add_argument('--edge_gate', type=str, choices=['none', 'proj', 'free'], default=edge_gate,
                        help='types of independent edge gate')
    parser.add_argument('--n_classes', type=int, default=num_classes, help='number of classes for the output layer')
    parser.add_argument('--nodemodel', type=str, default=nodemodel, choices=['additive', 'mlp', 'attention'],
                        help='name of node model class')
    parser.add_argument('--final', type=str, default=final, choices=['none', 'proj'], help='final output layer')
    # attention
    parser.add_argument('--nheads', type=int, nargs='*', default=nheads, help='number of heads in multihead attention')
    parser.add_argument('--att_act', type=str, default=att_act, choices=['none', 'lrelu', 'relu', 'elu'],
                        help='attention activation function in multihead attention')
    parser.add_argument('--att_dropout', type=float, default=att_dropout,
                        help='attention dropout in multihead attention')
    parser.add_argument('--att_dir', type=str, default=att_dir, choices=['in', 'out'],
                        help='attention direction in multihead attention')
    parser.add_argument('--att_combine', type=str, default=att_combine, choices=['cat', 'add', 'mean'],
                        help='multihead combination method in multihead attention')
    parser.add_argument('--att_combine_out', type=str, default=att_combine, choices=['cat', 'add', 'mean'],
                        help='multihead combination method in multihead attention for the last output attention layer')
    # other model arguments
    parser.add_argument('--deg_norm', type=str, choices=['none', 'sm', 'rw'], default=deg_norm,
                        help='degree normalization method')
    parser.add_argument('--aggr', type=str, choices=['add', 'mean', 'max'], default=aggr,
                        help='feature aggregation method')
    parser.add_argument('--dropout', type=float, default=dropout, help='dropout probability')
    parser.add_argument('--bias', type=int, default=bias, help='whether to include bias in the model')
    # optimization
    parser.add_argument('--lr', type=float, default=learning_rate, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--epochs', type=int, default=num_epochs, help='number of training epochs')
    parser.add_argument('--early_stop', type=int, default=early_stop, help='whether to do early stopping')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='directory to save the best model')
    parser.add_argument('--save_name', type=str, default=save_name, help='file name to save the best model')
    args = parser.parse_args()
    return args


def train(model, args, train_loader, val_dataset, test_dataset, optimizer, criterion,
          scheduler=None, logger=None):
    if logger is None:
        logging = print
    else:
        logging = logger.info

    device = next(model.parameters()).device
    predictor = PygModelPredictor(model)

    early_stopper = EarlyStopping(patience=5, mode='min', verbose=True, logger=logger)

    best_epoch = 0
    start = time.time()
    for ep in range(args.epochs):
        loss_avg_train = 0
        num_train_graph = 0
        model.train()
        for n, batch in enumerate(train_loader):
            batch.to(device)

            optimizer.zero_grad()

            x = model(batch.x, batch.edge_index)
            loss = criterion(x, batch.y.long())

            loss_avg_train += float(loss)
            num_train_graph += batch.num_graphs

            loss.backward()
            optimizer.step()

            if num_train_graph % args.log_interval == 0 or n == len(train_loader) - 1:
                with torch.no_grad():
                    # pred = x.argmax(dim=1)
                    pred_prob = torch.softmax(x, dim=1)[:, 1]
                    y = batch.y.long()
                    result_dict = eval_metrics(y, pred_prob)
                logging(f'epoch: {ep + 1}, passed number of graphs: {num_train_graph}, '
                        f'train running loss: {loss_avg_train / num_train_graph:.5f} (passed time: {time_since(start)})')
                logging(' ' * 10 + ', '.join(['{}: {:.5f}'.format(k, v) for k, v in result_dict.items()]))

        result_dict_avg, loss_avg = eval_predictor(val_dataset, predictor)
        logging(f'Validation --- epoch: {ep + 1}, loss: {loss_avg:.5f}')
        logging(' ' * 10 + ', '.join(['{}: {:.5f}'.format(k, v) for k, v in result_dict_avg.items()]))

        if scheduler is not None:
            scheduler.step(loss_avg)

        if args.early_stop:
            early_stopper(loss_avg)
        else:
            early_stopper.improved = True

        if early_stopper.improved:
            torch.save(model, os.path.join(args.save_dir, args.save_name))
            logging(f'model saved at {os.path.join(args.save_dir, args.save_name)}.')
            best_epoch = ep
        elif early_stopper.early_stop:
            logging(f'Early stopping here.')
            break
        else:
            pass

    if early_stopper.improved:
        best_model = model
    else:
        best_model = torch.load(os.path.join(args.save_dir, args.save_name))
    logging('*' * 12 + f' best model obtained after epoch {best_epoch + 1}, '
                       f'saved at {os.path.join(args.save_dir, args.save_name)} ' + '*' * 12)
    predictor = PygModelPredictor(best_model)
    result_dict_avg, loss_avg = eval_predictor(test_dataset, predictor)
    logging(f'Testing --- loss: {loss_avg:.5f}')
    logging(' ' * 10 + ', '.join(['{}: {:.5f}'.format(k, v) for k, v in result_dict_avg.items()]))


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== random seeds and device
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = torch.device(f'cuda:{args.devid}') if args.devid > -1 else torch.device('cpu')

    # ========== logging setup
    log_name = os.path.splitext(args.save_name)[0]
    logger = logging_config(__name__, folder=args.save_dir, name=log_name, filemode=args.logmode)
    # logger = logging_config(os.path.basename(__file__), folder=args.save_dir, name=log_name, filemode=args.logmode)

    logger.info('python ' + ' '.join(sys.argv))
    logger.info('-' * 30)
    logger.info(args)
    logger.info('-' * 30)
    logger.info(time.ctime())
    logger.info('-' * 30)

    # ========== load the dataset
    logger.info('loading dataset...')

    train_ds = BotnetDataset(name=args.data_name, root=args.data_dir, split='train',
                             in_memory=bool(args.in_memory), graph_format='pyg')
    val_ds = BotnetDataset(name=args.data_name, root=args.data_dir, split='val',
                             in_memory=bool(args.in_memory), graph_format='pyg')
    test_ds = BotnetDataset(name=args.data_name, root=args.data_dir, split='test',
                             in_memory=bool(args.in_memory), graph_format='pyg')

    train_loader = GraphDataLoader(train_ds, batch_size=args.batch_size, shuffle=bool(args.shuffle), num_workers=0)

    # ========== define the model, optimizer, and loss
    if len(args.nheads) < len(args.enc_sizes):
        assert len(args.nheads) == 1
        args.nheads = args.nheads * len(args.enc_sizes)
    elif len(args.nheads) == len(args.enc_sizes):
        pass
    else:
        raise ValueError

    final_layer_config = {'att_combine': args.att_combine_out}

    model = GCNModel(args.in_channels,
                     args.enc_sizes,
                     args.n_classes,
                     non_linear=args.act,
                     non_linear_layer_wise=args.layer_act,
                     residual_hop=args.residual_hop,
                     dropout=args.dropout,
                     final_layer_config=final_layer_config,
                     final_type=args.final,
                     pred_on='node',
                     nodemodel=args.nodemodel,
                     deg_norm=args.deg_norm,
                     edge_gate=args.edge_gate,
                     aggr=args.aggr,
                     bias=bool(args.bias),
                     nheads=args.nheads,
                     att_act=args.att_act,
                     att_dropout=args.att_dropout,
                     att_dir=args.att_dir,
                     att_combine=args.att_combine,
                     )

    logger.info('model ' + '-' * 10)
    logger.info(repr(model))
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1)

    # ========== train the model
    train(model, args, train_loader, val_ds, test_ds, optimizer, criterion,
          scheduler, logger)
