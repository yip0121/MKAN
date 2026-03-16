import copy
import json
import os

from utils.metrics import metric


COMMON_SPACE = {
    'learning_rate': ('log_float', 1e-4, 5e-3),
    'batch_size': ('categorical', [16, 32, 64]),
    'dropout': ('float', 0.0, 0.3),
}

MODEL_SEARCH_SPACE = {
    'TimeKAN': {
        **COMMON_SPACE,
        'd_model': ('categorical', [8, 16, 32, 64]),
        'e_layers': ('int', 1, 4),
        'd_ff': ('categorical', [16, 32, 64, 128]),
        'begin_order': ('int', 1, 4),
        'down_sampling_layers': ('int', 0, 2),
    },
    'CNNLSTM': {
        **COMMON_SPACE,
        'cnn_channels': ('categorical', [8, 16, 32, 64]),
        'lstm_hidden': ('categorical', [16, 32, 64, 128]),
        'lstm_layers': ('int', 1, 3),
    },
    'iTransformer': {
        **COMMON_SPACE,
        'd_model': ('categorical', [16, 32, 64, 128]),
        'n_heads': ('categorical', [1, 2, 4, 8]),
        'itf_layers': ('int', 1, 4),
        'd_ff': ('categorical', [32, 64, 128, 256]),
    },
    'TimeMixer': {
        **COMMON_SPACE,
        'd_model': ('categorical', [16, 32, 64, 128]),
        'time_mixer_layers': ('int', 1, 4),
        'time_mixer_kernel': ('categorical', [3, 5, 7]),
    },
}


def build_bayes_best_filename(base_args):
    train_tag = int(round(float(base_args.train_ratio) * 100))
    model = str(base_args.model)
    return f'bayes_opt_best_{model}_train{train_tag}.json'


def _suggest(trial, name, spec):
    stype = spec[0]
    if stype == 'log_float':
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if stype == 'float':
        return trial.suggest_float(name, spec[1], spec[2])
    if stype == 'int':
        return trial.suggest_int(name, spec[1], spec[2])
    if stype == 'categorical':
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f'Unknown search type: {stype}')


def _resolve_search_space(model_name):
    if model_name not in MODEL_SEARCH_SPACE:
        raise ValueError(f"No Bayesian search space configured for model '{model_name}'. Supported: {list(MODEL_SEARCH_SPACE.keys())}")
    return MODEL_SEARCH_SPACE[model_name]


def run_bayesian_optimization(base_args, ExpClass, build_setting_name_fn):
    """Run Optuna (TPE) Bayesian optimization and return updated args."""
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError(
            'Bayesian optimization requested but optuna is unavailable. '
            'Please `pip install optuna` or disable --enable_bayes_opt.'
        ) from exc

    search_space = _resolve_search_space(base_args.model)
    print(f'[BayesOpt] Active model: {base_args.model}')
    print('[BayesOpt] Search space:')
    print(json.dumps(search_space, indent=2, default=str))

    def objective(trial):
        args_trial = copy.deepcopy(base_args)

        sampled = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}
        for k, v in sampled.items():
            setattr(args_trial, k, v)

        args_trial.train_epochs = base_args.bayes_train_epochs
        args_trial.patience = min(base_args.patience, max(3, base_args.bayes_train_epochs // 2))
        args_trial.itr = 1
        args_trial.comment = f'bayes_t{trial.number}'
        args_trial.model_id = f'{base_args.model_id}_bo'

        setting = build_setting_name_fn(args_trial, 0)
        exp = ExpClass(args_trial)
        exp.train(setting, save_checkpoint=False)

        val_data, _ = exp._get_data(flag='val')
        exp.model.eval()
        if args_trial.pred_len == 1:
            _, val_loader = exp._get_data(flag='val')
            preds_q, trues = exp._test_single_step_loader(val_loader)
        else:
            preds_q, trues = exp._test_multi_step_direct(val_data)

        preds = preds_q[:, :, exp.q_median_idx:exp.q_median_idx + 1]
        _, mse, _, _, _ = metric(preds, trues)
        mse = float(mse)

        trial.set_user_attr('setting', setting)
        trial.set_user_attr('sampled', sampled)
        print(f"[BayesOpt][Trial {trial.number}] setting={setting} val_mse={mse:.6f} params={sampled}")
        return mse

    sampler = optuna.samplers.TPESampler(seed=2021)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    timeout = None if base_args.bayes_timeout <= 0 else base_args.bayes_timeout
    study.optimize(objective, n_trials=base_args.bayes_trials, timeout=timeout)

    print('[BayesOpt] Best value (MSE):', study.best_value)
    print('[BayesOpt] Best params:', study.best_params)

    model_result_root = os.path.join(base_args.project_root, 'results', base_args.model)
    os.makedirs(model_result_root, exist_ok=True)
    best_path = os.path.join(model_result_root, build_bayes_best_filename(base_args))

    best_payload = {
        'model': base_args.model,
        'objective_split': 'val',
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_setting': study.best_trial.user_attrs.get('setting', ''),
        'n_trials': len(study.trials),
    }
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(best_payload, f, indent=2)

    print(f'[BayesOpt] Saved best summary to: {best_path}')

    tuned_args = copy.deepcopy(base_args)
    for k, v in study.best_params.items():
        setattr(tuned_args, k, v)

    tuned_args.comment = f'{base_args.comment}_bayes_best'
    return tuned_args, best_payload
