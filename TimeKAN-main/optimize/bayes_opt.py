import copy
import json
import os

import numpy as np
import pandas as pd


# Reasonable high-impact search space for current SOH workflow.
SEARCH_SPACE = {
    'learning_rate': ('log_float', 1e-4, 5e-3),
    'batch_size': ('categorical', [16, 32, 64]),
    'd_model': ('categorical', [8, 16, 32, 64]),
    'e_layers': ('int', 1, 4),
    'd_ff': ('categorical', [16, 32, 64, 128]),
    'dropout': ('float', 0.0, 0.3),
    'begin_order': ('int', 1, 4),
    'down_sampling_layers': ('int', 0, 2),
}


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


def run_bayesian_optimization(base_args, ExpClass, build_setting_name_fn):
    """Run Optuna (TPE) Bayesian optimization and return updated args.

    Objective: minimize MSE from saved metrics.npy after each trial.
    """
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError(
            'Bayesian optimization requested but optuna is unavailable. '\
            'Please `pip install optuna` or disable --enable_bayes_opt.'
        ) from exc

    print('[BayesOpt] Search space:')
    print(json.dumps(SEARCH_SPACE, indent=2, default=str))

    def objective(trial):
        args_trial = copy.deepcopy(base_args)

        sampled = {name: _suggest(trial, name, spec) for name, spec in SEARCH_SPACE.items()}
        for k, v in sampled.items():
            setattr(args_trial, k, v)

        # Use shorter inner-loop training for optimization speed and stability.
        args_trial.train_epochs = base_args.bayes_train_epochs
        args_trial.patience = min(base_args.patience, max(3, base_args.bayes_train_epochs // 2))
        args_trial.itr = 1
        args_trial.comment = f'bayes_t{trial.number}'
        args_trial.model_id = f'{base_args.model_id}_bo'

        setting = build_setting_name_fn(args_trial, 0)
        exp = ExpClass(args_trial)
        exp.train(setting)
        exp.test(setting)

        metrics_path = os.path.join('results', setting, 'metrics.npy')
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f'Expected metrics file missing: {metrics_path}')

        metrics = np.load(metrics_path)
        mse = float(metrics[1])
        trial.set_user_attr('setting', setting)
        trial.set_user_attr('sampled', sampled)
        print(f"[BayesOpt][Trial {trial.number}] setting={setting} mse={mse:.6f} params={sampled}")
        return mse

    sampler = optuna.samplers.TPESampler(seed=2021)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    timeout = None if base_args.bayes_timeout <= 0 else base_args.bayes_timeout
    study.optimize(objective, n_trials=base_args.bayes_trials, timeout=timeout)

    print('[BayesOpt] Best value (MSE):', study.best_value)
    print('[BayesOpt] Best params:', study.best_params)

    # Persist study summary and all trial records
    os.makedirs('results', exist_ok=True)
    best_path = os.path.join('results', 'bayes_opt_best.json')
    trials_path = os.path.join('results', 'bayes_opt_trials.csv')

    best_payload = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_setting': study.best_trial.user_attrs.get('setting', ''),
        'n_trials': len(study.trials),
    }
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(best_payload, f, indent=2)

    trial_rows = []
    for t in study.trials:
        row = {
            'trial': t.number,
            'value_mse': t.value,
            'state': str(t.state),
            'setting': t.user_attrs.get('setting', ''),
        }
        row.update(t.params)
        trial_rows.append(row)
    pd.DataFrame(trial_rows).to_csv(trials_path, index=False)

    print(f'[BayesOpt] Saved best summary to: {best_path}')
    print(f'[BayesOpt] Saved trial table to: {trials_path}')

    tuned_args = copy.deepcopy(base_args)
    for k, v in study.best_params.items():
        setattr(tuned_args, k, v)

    tuned_args.comment = f'{base_args.comment}_bayes_best'
    return tuned_args
