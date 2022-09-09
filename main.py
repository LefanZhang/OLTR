import os
import argparse
import pprint
from data import dataloader
#from run_networks import model
from run_networks_for_CoMix import model
import warnings
from utils import source_import

import torch
import numpy as np
import random


# random seeds
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# ================
# LOAD CONFIGURATIONS

data_root = {#'ImageNet': '/home/public/public_dataset/ILSVRC2014/Img',
             'ImageNet': '/data1/ImageNet',
             'Places': '/home/public/dataset/Places365'}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/Imagenet_LT/Stage_1.py', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--gpu', default=False, type=int)
parser.add_argument('--trial', default=1, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits

config = source_import(args.config).config
config['training_opt']['log_dir'] += '/trial_{}'.format(args.trial) # decide which experiment it is
training_opt = config['training_opt']
# change
relatin_opt = config['memory']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
pprint.pprint(config)

if not test_mode:

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        sampler_dic = {'sampler': source_import(sampler_defs['def_file']).get_sampler(), 
                       'num_samples_cls': sampler_defs['num_samples_cls']}
    else:
        sampler_dic = None

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x, 
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic,
                                    num_workers=training_opt['num_workers'])
            #for x in (['train', 'val', 'train_plain'] if relatin_opt['init_centroids'] else ['train', 'val'])}
            for x in (['train', 'val', 'train_plain'] if relatin_opt['init_prototypes'] else ['train', 'val'])}

    training_model = model(config, data, test=False)

    training_model.train_for_CoMix()

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None, 
                                    test_open=test_open,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=False)
            for x in ['train', 'test']}

    
    training_model = model(config, data, test=True)
    training_model.load_model()
    # training_model.eval_for_CoMix(phase='test', openset=test_open)
    training_model.eval_with_prototypes(phase='test', openset=test_open)

    if output_logits:
        training_model.output_logits(openset=test_open)
        
print('ALL COMPLETED.')
