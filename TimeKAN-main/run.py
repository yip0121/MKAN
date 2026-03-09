import argparse
import os
import random
import re

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from optimize.bayes_opt import run_bayesian_optimization


def seed_everything(seed: int = 2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_parser():
    parser = argparse.ArgumentParser(description='TimeKAN for Battery SOH forecasting')

    # task
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='battery_soh')
    parser.add_argument('--model', type=str, default='TimeKAN')

    # battery soh data
    parser.add_argument('--data', type=str, default='battery_soh')
    parser.add_argument('--root_path', type=str, default='./dataset/battery/')
    parser.add_argument('--data_path', type=str, default='battery_36Ah_70W_65W_1551.xlsx')
    parser.add_argument('--target', type=str, default='soh')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation split ratio')

    # sequence
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--quantiles', type=str, default='0.95,0.5,0.05', help='comma-separated quantiles for pinball loss, e.g. 0.95,0.5,0.05')

    # model
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--begin_order', type=int, default=1)
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--use_future_temporal_feature', type=int, default=0)

    # training
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lradj', type=str, default='TST')
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true', default=False)

    # bayesian optimization
    parser.add_argument('--enable_bayes_opt', action='store_true', default=False, help='run Bayesian hyper-parameter optimization before final training')
    parser.add_argument('--bayes_trials', type=int, default=15, help='number of Bayesian optimization trials')
    parser.add_argument('--bayes_train_epochs', type=int, default=8, help='epochs per Bayesian trial')
    parser.add_argument('--bayes_timeout', type=int, default=0, help='timeout seconds for Bayesian optimization; <=0 means no timeout')
    parser.add_argument('--bayes_refit', action='store_true', default=False, help='after Bayesian optimization, refit once with best parameters')

    # misc kept for compatibility with existing experiment naming
    parser.add_argument('--comment', type=str, default='none')
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--output_attention', action='store_true', default=False)
    parser.add_argument('--inverse', action='store_true', default=False)

    # gpu
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1')

    # internal path control (auto-filled in main)
    parser.add_argument('--project_root', type=str, default='')

    return parser



def parse_quantiles(quantiles_str):
    quantiles = [float(q.strip()) for q in quantiles_str.split(',') if q.strip() != '']
    if len(quantiles) == 0:
        raise ValueError('quantiles must not be empty')
    for q in quantiles:
        if q <= 0 or q >= 1:
            raise ValueError(f'quantile must be in (0,1), got {q}')
    return quantiles

def validate_split_args(args):
    if args.train_ratio <= 0 or args.val_ratio < 0:
        raise ValueError('train_ratio must be > 0 and val_ratio must be >= 0')
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError('train_ratio + val_ratio must be < 1, leaving space for test set')




def build_dataset_size_tag(data_path):
    base = os.path.basename(data_path)
    m = re.search(r'(\d+Ah)', base, re.IGNORECASE)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]

def build_setting_name(args, ii):
    """Keep result/checkpoint folder naming compatible with the previous format."""
    return '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.comment,
        args.model,
        build_dataset_size_tag(args.data_path),
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii
    )

def main():
    seed_everything(2021)
    parser = build_parser()
    args = parser.parse_args()
    args.project_root = os.path.dirname(os.path.abspath(__file__))
    validate_split_args(args)
    args.quantiles = parse_quantiles(args.quantiles)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if os.name == 'nt' and args.num_workers > 0:
        print('[Info] Windows detected: overriding --num_workers to 0 for stability.')
        args.num_workers = 0

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    test_ratio = 1 - args.train_ratio - args.val_ratio
    print(f'Data split => train: {args.train_ratio:.2f}, val: {args.val_ratio:.2f}, test: {test_ratio:.2f}')
    print('Project root:', args.project_root)
    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast

    if args.enable_bayes_opt:
        print('[BayesOpt] Starting Bayesian hyper-parameter optimization...')
        tuned_args = run_bayesian_optimization(args, Exp, build_setting_name)
        print('[BayesOpt] Best params injected into runtime args.')
        args = tuned_args
        print('Args after BayesOpt:')
        print(args)
        if not args.bayes_refit:
            print('[BayesOpt] bayes_refit is False; skipping final retraining.')
            return

    if args.is_training:
        for ii in range(args.itr):
            setting = build_setting_name(args, ii)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = build_setting_name(args, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
