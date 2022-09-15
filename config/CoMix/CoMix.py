# Testing configurations
config = {}

training_opt = {}
training_opt['dataset'] = 'ImageNet_LT'
training_opt['log_dir'] = './logs/CoMix/stage1'
training_opt['num_classes'] = 1000
training_opt['batch_size'] = 128
training_opt['num_workers'] = 8
training_opt['num_epochs'] = 90
training_opt['display_step'] = 10
training_opt['feature_dim'] = 512
training_opt['open_threshold'] = 0.1
training_opt['sampler'] = None
training_opt['scheduler_params'] = {'step_size': 30, 'gamma': 0.1}  # lr decay
training_opt['schedule_loss_weight'] = False
training_opt['eval_with_prototypes'] = 1    # 0 for max, 1 for avg
training_opt['discriminative_feature_space'] = True
training_opt['balanced_feature_space'] = False
training_opt['aug_for_psc'] = 1 # 0 for sim-sim, 1 for sim-rand, 2 for randstack-randstack, 3 for no aug for psc
training_opt['which_aug_to_use'] = [1,2,3]    # 1 for aug1, 2 for aug2, 3 for aug3
training_opt['pretrain'] = 60   # apply PSC Loss after 30 epochs
training_opt['reinit'] = 0  # 0 for avg, 1 for replace, 2 for using init
config['training_opt'] = training_opt

networks = {}
feature_param = {'use_modulatedatt': False, 'use_fc': False, 'dropout': None, 'use_mlp': True,  # use_fc: append a fc+relu after avgpool; use_mlp: transform feature for contrastive learning
                 'stage1_weights': False, 'dataset': training_opt['dataset']}
feature_optim_param = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0005}
networks['feat_model'] = {'def_file': './models/ResNet10Feature.py',
                          'params': feature_param,
                          'optim_params': feature_optim_param,
                          'fix': False}
classifier_param = {'in_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes'], 'scale': 16., # scale default=16
                    'stage1_weights': False, 'dataset': training_opt['dataset']}
classifier_optim_param = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0005}
networks['classifier'] = {'def_file': './models/CosNormClassifier_for_CoMix.py',
                          'params': classifier_param,
                          'optim_params': classifier_optim_param}
config['networks'] = networks

criterions = {}
perf_loss_param = {}
# criterions['PerformanceLoss'] = {'def_file': './loss/SoftmaxLoss.py', 'loss_params': perf_loss_param,
#                                  'optim_params': None, 'weight': 1.0}
criterions['PerformanceLoss'] = {'def_file': './loss/BalancedSoftmaxLoss.py', 'loss_params': perf_loss_param,
                                 'optim_params': None, 'weight': 1.0}


# feat_loss_param = {'feat_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes']}
# feat_loss_optim_param = {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0005}
# criterions['FeatureLoss'] = {'def_file': './loss/DiscCentroidsLoss.py', 'loss_params': feat_loss_param,
#                              'optim_params': feat_loss_optim_param, 'weight': 0.01}


psc_loss_param = {'temp': 0.1}
criterions['PSCLoss'] = {'def_file': './loss/PSCLoss.py', 'loss_params': psc_loss_param,
                        'optim_params': None, 'weight': 1.0}

config['criterions'] = criterions

memory = {}
# memory['centroids'] = False
# memory['init_centroids'] = False
memory['prototypes'] = True
memory['init_prototypes'] = True
memory['prototypes_num'] = 4
memory['update'] = 0    # 0 for update_prototypes, 1 for update_prototypes_new
memory['update_mode'] = 1   # effective when update=1, 0 for use features with right prediction, 1 for use all
memory['normal_before_add'] = True  # effective when update=1
memory['ema'] = 0.9 # default 0.9
memory['std'] = 0.1 # default 0.1
config['memory'] = memory
