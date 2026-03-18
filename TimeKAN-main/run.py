import argparse
import json
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

    def str2bool(value):
        if isinstance(value, bool):
            return value
        normalized = value.strip().lower()
        if normalized in {'true', '1', 'yes', 'y'}:
            return True
        if normalized in {'false', '0', 'no', 'n'}:
            return False
        raise argparse.ArgumentTypeError(f'Expected a boolean value, got: {value}')

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
    parser.add_argument('--prediction_target', type=str, default='absolute', choices=['absolute', 'delta'],
                        help='train/test target type: absolute SOH or first-order delta SOH')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation split ratio')

    # sequence
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--multi_step_stride', type=int, default=1,
                        help='sliding stride for multi-step direct testing; use 1 for dense last-step evaluation')
    parser.add_argument('--quantiles', type=str, default='0.95,0.5,0.05', help='comma-separated quantiles for pinball loss, e.g. 0.95,0.5,0.05')
    parser.add_argument('--eval_last_step_only', type=str2bool, nargs='?', const=True, default=True,
                        help='if True, only evaluate/plot the final step of each forecast window')
    parser.add_argument('--clip_predictions', type=str2bool, nargs='?', const=True, default=True,
                        help='clip restored absolute predictions to a plausible SOH range for stable metrics/plots')
    parser.add_argument('--clip_margin', type=float, default=2.0,
                        help='margin added to observed test SOH min/max when clipping predictions')

    # model
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=1)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--begin_order', type=int, default=1)
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--use_future_temporal_feature', type=int, default=0)
    parser.add_argument('--wavelet', type=str, default='haar')
    parser.add_argument('--wavelet_mode', type=str, default='symmetric')
    parser.add_argument('--dwt_level', type=int, default=3)

    # training
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lradj', type=str, default='TST')
    parser.add_argument('--pct_start', type=float, default=0.2)
    default_workers = max(1, min(8, (os.cpu_count() or 1) // 2))
    parser.add_argument('--num_workers', type=int, default=default_workers,
                        help='DataLoader worker processes. Default auto-tuned from CPU cores (1~8).')
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'pinball'],
                        help='training loss type: mse (default) or pinball quantile loss')

    # bayesian optimization
    parser.add_argument('--enable_bayes_opt', action='store_true', default=False, help='run Bayesian hyper-parameter optimization before final training')
    parser.add_argument('--disable_bayes_opt', action='store_false', dest='enable_bayes_opt', help='disable Bayesian hyper-parameter optimization')
    parser.add_argument('--bayes_trials', type=int, default=30, help='number of Bayesian optimization trials')
    parser.add_argument('--bayes_train_epochs', type=int, default=30, help='epochs per Bayesian trial')
    parser.add_argument('--bayes_timeout', type=int, default=0, help='timeout seconds for Bayesian optimization; <=0 means no timeout')
    parser.add_argument('--bayes_refit', action='store_true', default=False, help='after Bayesian optimization, refit once with best parameters')
    parser.add_argument('--no_bayes_refit', action='store_false', dest='bayes_refit', help='skip final retraining after Bayesian optimization')

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
    parser.add_argument('--use_gpu', type=str2bool, nargs='?', const=True, default=True)
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
    if args.multi_step_stride <= 0:
        raise ValueError('multi_step_stride must be a positive integer')




def build_dataset_size_tag(data_path):
    base = os.path.basename(data_path)
    m = re.search(r'(\d+Ah)', base, re.IGNORECASE)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]

def build_setting_name(args, ii):
    """Keep result/checkpoint folder naming compatible with the previous format."""
    return '{}_{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.train_ratio,
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




def read_metrics_csv(metrics_path):
    import pandas as pd
    metrics_df = pd.read_csv(metrics_path)
    required_cols = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'r2']
    if metrics_df.empty or not all(col in metrics_df.columns for col in required_cols):
        return None
    row = metrics_df.iloc[0]
    return {col: float(row[col]) for col in required_cols}




def build_bayes_best_path(args):
    train_tag = int(round(float(args.train_ratio) * 100))
    return os.path.join(args.project_root, 'results', f'bayes_opt_best_train{train_tag}.json')

def update_bayes_json_with_refit(args, refit_setting):
    best_path = build_bayes_best_path(args)
    refit_metrics_path = os.path.join(args.project_root, 'results', refit_setting, 'metrics.csv')
    if not os.path.exists(best_path) or not os.path.exists(refit_metrics_path):
        return

    with open(best_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    refit_metrics = read_metrics_csv(refit_metrics_path)
    if refit_metrics is None:
        return

    payload['refit_setting'] = refit_setting
    payload['refit_metrics'] = refit_metrics

    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f'[BayesOpt] Updated best summary with refit metrics: {best_path}')

def print_result_summary(args, setting, header='Result Summary'):
    metrics_path = os.path.join(args.project_root, 'results', setting, 'metrics.csv')
    fig_path = os.path.join(args.project_root, 'results', setting, 'prediction_vs_truth_with_interval.png')
    csv_path = os.path.join(args.project_root, 'results', setting, 'prediction_vs_truth.csv')

    print(f'[{header}] setting: {setting}')
    if os.path.exists(metrics_path):
        metrics = read_metrics_csv(metrics_path)
        if metrics is not None:
            print(f'[{header}] mae={metrics["mae"]:.6f}, mse={metrics["mse"]:.6f}, rmse={metrics["rmse"]:.6f}, mape={metrics["mape"]:.6f}, mspe={metrics["mspe"]:.6f}, r2={metrics["r2"]:.6f}')
        else:
            print(f'[{header}] metrics.csv exists but format unexpected: {metrics_path}')
    else:
        print(f'[{header}] metrics not found: {metrics_path}')

    print(f'[{header}] interval figure: {fig_path}')
    print(f'[{header}] csv: {csv_path}')

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
        tuned_args, best_info = run_bayesian_optimization(args, Exp, build_setting_name)
        print('[BayesOpt] Best params injected into runtime args.')
        print('[BayesOpt] Final best parameter configuration:', best_info.get('best_params', {}))
        print('[BayesOpt] Final best setting:', best_info.get('best_setting', ''))
        args = tuned_args
        print('Args after BayesOpt:')
        print(args)
        if not args.bayes_refit:
            print('[BayesOpt] bayes_refit is False; skipping final retraining.')
            print('[BayesOpt] Best trial is selected by validation MSE only; test set is untouched in Bayesian search.')
            print(f"[BayesOpt] Best validation MSE: {best_info.get('best_value', 'N/A')}")
            return
        print('[BayesOpt] bayes_refit is True; final training metrics can differ from best trial metrics due to more epochs/retraining.')

    if args.is_training:
        for ii in range(args.itr):
            setting = build_setting_name(args, ii)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, save_dwt_plot=(ii == args.itr - 1))
            print_result_summary(args, setting, header='Final Training Run')
            if args.enable_bayes_opt and args.bayes_refit and ii == 0:
                update_bayes_json_with_refit(args, setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = build_setting_name(args, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1, save_dwt_plot=True)
        print_result_summary(args, setting, header='Test-Only Run')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
