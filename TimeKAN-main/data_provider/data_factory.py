from data_provider.data_loader import Dataset_BatterySOH
from torch.utils.data import DataLoader


def data_provider(args, flag):
    if args.data != 'battery_soh':
        raise ValueError(f"This streamlined pipeline only supports '--data battery_soh', got: {args.data}")

    shuffle_flag = flag != 'test'
    drop_last = True

    data_set = Dataset_BatterySOH(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
