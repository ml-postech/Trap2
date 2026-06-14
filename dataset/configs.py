from .eurosat import EuroSATBase
from .cars import Cars
from .dtd import DTD
from .mnist import MNIST
from .gtsrb import GTSRB
from .svhn import SVHN
from .resisc45 import RESISC45
from .fgvc_aircraft import FGVCAircraft




eurosat = {
    'wrapper': EuroSATBase,
    'batch_size': 32,
    'res': 224,
    'type': 'eurosat',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/eurosat'
}

stanford_cars = {
    'wrapper': Cars,
    'batch_size': 32,
    'res': 224,
    'type': 'stanford_cars',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/stanford_cars'
}

mnist = {
    'wrapper': MNIST,
    'batch_size': 32,
    'res': 224,
    'type': 'mnist',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/mnist'
}

svhn = {
    'wrapper': SVHN,
    'batch_size': 32,
    'res': 224,
    'type': 'svhn',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/svhn'
}

dtd = {
    'wrapper': DTD,
    'batch_size': 32,
    'res': 224,
    'type': 'dtd',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/dtd'
}


gtsrb = {
    'wrapper': GTSRB,
    'batch_size': 32,
    'res': 224,
    'type': 'gtsrb',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/gtsrb'
}

resisc45 = {
    'wrapper': RESISC45,
    'batch_size': 32,
    'res': 224,
    'type': 'resisc45',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/resisc45'
}

fgvc_aircraft = {
    'wrapper': FGVCAircraft,
    'batch_size': 32,
    'res': 224,
    'type': 'fgvc_aircraft',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/fgvc_aircraft',
    'target_type': 'variant'
}

# Math generative tasks
gsm8k = {
    'type': 'gsm8k',
    'batch_size': 4,
    'num_workers': 8,
    'model_name_or_path': 'meta-llama/Llama-3.2-1B',
    'max_length': 512,
    'val_fraction': 0.1,
}

asdiv = {
    'type': 'asdiv',
    'batch_size': 4,
    'num_workers': 8,
    'model_name_or_path': 'meta-llama/Llama-3.2-1B',
    'max_length': 512,
    'val_fraction': 0.0,  # ASDiv splits manually in MathDataset
}

