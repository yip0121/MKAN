import copy
import json
import os

from utils.metrics import metric


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




def build_bayes_best_filename(base_args):
    train_tag = int(round(float(base_args.train_ratio) * 100))
    return f'bayes_opt_best_train{train_tag}.json'

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

    Objective: minimize validation-set MSE for each trial without using test set.
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
        exp.train(setting, save_checkpoint=False)

        val_data, _ = exp._get_data(flag='val')
        exp.model.eval()
        if args_trial.pred_len == 1:
            _, val_loader = exp._get_data(flag='val')
            preds_q, trues = exp._test_single_step_loader(val_loader)
        else:
            preds_q, trues = exp._test_multi_step_stride1_laststep(val_data)

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

    # Persist best summary only (as requested)
    os.makedirs(os.path.join(base_args.project_root, 'results'), exist_ok=True)
    best_path = os.path.join(base_args.project_root, 'results', build_bayes_best_filename(base_args))

    best_payload = {
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
