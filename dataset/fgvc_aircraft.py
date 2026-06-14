import torch
import torchvision.datasets as datasets


ROOT = "data"  # Path to the root directory of the dataset


class FGVCAircraft:
    def __init__(self,
                 is_train,
                 preprocess,
                 location=ROOT,
                 batch_size=128,
                 num_workers=16,
                 target_type="variant"):
        split = "train" if is_train else "test"
        try:
            self.dataset = datasets.FGVCAircraft(
                root=location,
                split=split,
                target_type=target_type,
                download=True,
                transform=preprocess,
            )
        except TypeError:
            self.dataset = datasets.FGVCAircraft(
                root=location,
                split=split,
                download=True,
                transform=preprocess,
            )

        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
        )

        if hasattr(self.dataset, "classes"):
            self.classnames = list(self.dataset.classes)
        else:
            idx_to_class = dict((v, k) for k, v in self.dataset.class_to_idx.items())
            self.classnames = [idx_to_class[i] for i in range(len(idx_to_class))]


def prepare_train_loaders(config):
    dataset_class = FGVCAircraft(
        is_train=True,
        preprocess=config['train_preprocess'],
        location=config.get('dir', ROOT),
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        target_type=config.get('target_type', "variant"),
    )
    loaders = {
        'full': dataset_class.loader
    }
    return loaders


def prepare_test_loaders(config):
    test_dataset_class = FGVCAircraft(
        is_train=False,
        preprocess=config['eval_preprocess'],
        location=config.get('dir', ROOT),
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        target_type=config.get('target_type', "variant"),
    )

    loaders = {
        'test': test_dataset_class.loader
    }

    try:
        val_dataset_class = FGVCAircraft(
            is_train=False,
            preprocess=config['eval_preprocess'],
            location=config.get('dir', ROOT),
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            target_type=config.get('target_type', "variant"),
        )
        val_dataset_class.dataset = datasets.FGVCAircraft(
            root=config.get('dir', ROOT),
            split="val",
            download=True,
            transform=config['eval_preprocess'],
        )
        val_dataset_class.loader = torch.utils.data.DataLoader(
            val_dataset_class.dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
        )
        loaders['val'] = val_dataset_class.loader
    except Exception:
        pass
    if config.get('val_fraction', 0) > 0.:
        print('splitting fgvc_aircraft')
        test_set = loaders['test'].dataset
        shuffled_idxs = torch.load(config['shuffled_idxs'], weights_only=False)
        num_valid = int(len(test_set) * config['val_fraction'])
        valid_idxs, test_idxs = shuffled_idxs[:num_valid], shuffled_idxs[num_valid:]
        val_set = torch.utils.data.Subset(test_set, valid_idxs)
        test_set = torch.utils.data.Subset(test_set, test_idxs)
        loaders['test'] = torch.utils.data.DataLoader(
            test_set,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
        loaders['val'] = torch.utils.data.DataLoader(
            val_set,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
    loaders['class_names'] = test_dataset_class.classnames

    return loaders
