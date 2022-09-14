import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import time
import numpy as np
import warnings
import pdb

class model ():
    
    def __init__(self, config, data, test=False, open=False):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.data = data
        self.test_mode = test
        
        # Initialize model
        self.init_models()

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            # If using steps for training, we need to calculate training steps 
            # for each epoch based on actual number of training data instead of 
            # oversampled data number 
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num  \
                                   / self.training_opt['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, \
            self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)

            
            if self.memory['init_prototypes']:   # initialize prototypes
                # self.criterions['FeatureLoss'].centroids.data = \
                #     self.centroids_cal(self.data['train_plain'])

                self.prototypes, self.sample_per_class = self.init_prototypes()    # not normalized
                # self.prototypes, self.sample_per_class = torch.randn(self.training_opt['num_classes'], self.memory['prototypes_num'], self.training_opt['feature_dim']).to(self.device), torch.ones(self.training_opt['num_classes'])
                print('Prototypes initialized!')
                print('Shape:', self.prototypes.shape)
                print('Initialized prototype visualization:', self.prototypes[:3])

            if 'Balanced' in self.config['criterions']['PerformanceLoss']['def_file']:
                self.config['criterions']['PerformanceLoss']['loss_params']['sample_per_class'] = self.sample_per_class
            self.init_criterions()

            
        # Set up log file
        if not self.test_mode:
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
        elif not open:
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log_test.txt')
        else:
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log_test_open.txt')
        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
    
    

    def init_prototypes(self):
        prototypes = torch.zeros(self.training_opt['num_classes'], self.training_opt['feature_dim'])
        class_count = torch.zeros(self.training_opt['num_classes'])


        bar = tqdm(total=self.epoch_steps+1)
        for step, (inputs, _, _, _, labels, _) in enumerate(self.data['train']):

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # X enable gradients
            with torch.set_grad_enabled(False):
                    
                features, _ = self.networks['feat_model'](inputs)
                #print(features.shape)
                for i, label in enumerate(labels):
                    prototypes[label] += features[i].cpu()
                    class_count[label] += 1

            bar.update(1)

        if 0 in class_count:
            print('Numerical failure during initializing prototypes!')
                
        prototypes /= class_count.view(-1, 1)

        # if self.memory['prototypes_num'] != 1:
        # print('Multi prototypes have not been implemented!')
        prototypes = prototypes.view(self.training_opt['num_classes'], 1, -1).repeat(1, self.memory['prototypes_num'], 1)
        # append noise
        prototypes += torch.randn(prototypes.shape) * self.memory['std']   # mean = 0, std = 0.1

        
        return prototypes.to(self.device), class_count


    def reinit_prototypes(self, mode=0):
        new_prototypes = torch.zeros(self.training_opt['num_classes'], self.memory['prototypes_num'], self.training_opt['feature_dim']).to(self.device)
        class_count = torch.zeros(self.training_opt['num_classes'], self.memory['prototypes_num']).to(self.device)


        bar = tqdm(total=self.epoch_steps+1)
        for step, (inputs, _, _, _, labels, _) in enumerate(self.data['train']):

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # X enable gradients
            with torch.set_grad_enabled(False):

                    
                features, _ = self.networks['feat_model'](inputs)
                #print(features.shape)
                for i, label in enumerate(labels):
                    _, which_to_update = F.normalize(self.prototypes[label], dim=1).matmul(features[i]).max(dim=0)
                    new_prototypes[label][which_to_update] += features[i]
                    class_count[label][which_to_update] += 1

            bar.update(1)

        if 0 in class_count:
            zero_of_each_class = [(line==0).sum() for line in class_count]
            print('Each class has some prototypes not reinited after pretrain:')
            print(zero_of_each_class)
            class_count = class_count.where(class_count==0, 1, class_count)
                
        new_prototypes /= class_count.view(-1, self.memory['prototypes_num'], 1)

        if mode == 1:
            return new_prototypes
        elif mode == 0:
            new_prototypes = (F.normalize(new_prototypes, dim=2) + F.normalize(self.prototypes, dim=2)) / 2 # avg
            return new_prototypes
        else:
            return None
        

        
    def init_models(self, optimizer=True):

        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        print("Using", torch.cuda.device_count(), "GPUs.")
        
        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            model_args = list(val['params'].values())
            model_args.append(self.test_mode)

            self.networks[key] = source_import(def_file).create_model(*model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).to(self.device)
            
            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for modulated attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'modulatedatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                 'lr': optim_params['lr'],
                                                 'momentum': optim_params['momentum'],
                                                 'weight_decay': optim_params['weight_decay']})

    def init_criterions(self):

        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = val['loss_params'].values()
            self.criterions[key] = source_import(def_file).create_loss(*loss_args).to(self.device)
            self.criterion_weights[key] = val['weight'] # weight to combine different losses
          
            if val['optim_params']: # optim the meta embeddings?
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.scheduler_params['step_size'],
                                              gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler

    def batch_forward (self, inputs, labels=None, centroids=False, feature_ext=False, phase='train'):
        '''
        This is a general single batch running function. 
        '''

        # Calculate Features
        self.features, self.feature_maps = self.networks['feat_model'](inputs)

        # If not just extracting features, calculate logits
        if not feature_ext:

            # During training, calculate centroids if needed to 
            if phase != 'test':
                if centroids and 'FeatureLoss' in self.criterions.keys():
                    self.centroids = self.criterions['FeatureLoss'].centroids.data
                else:
                    self.centroids = None

            # Calculate logits with classifier
            self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, self.centroids)



    def batch_forward_for_CoMix(self, inputs, labels=None, feature_ext=False, phase='train', epoch=None):
        '''
        This is a general single batch running function. 
        '''

        # Calculate Features
        self.features, self.feat_mlp = self.networks['feat_model'](inputs)

        # If not just extracting features, calculate logits
        if not feature_ext:

            # During training, calculate centroids if needed to 
            if phase != 'test':
                # if centroids and 'FeatureLoss' in self.criterions.keys():
                #     self.centroids = self.criterions['FeatureLoss'].centroids.data
                # else:
                #     self.centroids = None


                # print(self.networks['feat_model'].module.mlp)
                # if self.memory['prototypes_num'] == 1:
                #     self.prototypes_mlp = self.networks['feat_model'].module.mlp(self.prototypes).detach()
                #     self.prototypes_mlp = F.normalize(self.prototypes_mlp, dim=1)   # normalize to 1
                # else:
                if epoch and epoch > self.training_opt['pretrain']:
                    self.prototypes_mlp = self.networks['feat_model'].module.mlp(self.prototypes.view(-1, self.training_opt['feature_dim'])).detach()
                    self.prototypes_mlp = F.normalize(self.prototypes_mlp, dim=1).view(self.training_opt['num_classes'], -1, self.training_opt['feature_dim'])   # normalize to 1

            # Calculate logits with classifier
            self.logits = self.networks['classifier'](self.features)

    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers
        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):

        # First, apply performance loss
        self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels) \
                    * self.criterion_weights['PerformanceLoss']

        # Add performance loss to total loss
        self.loss = self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat

    def batch_loss_for_CoMix(self, labels, epoch):

        # First, apply performance loss
        self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels) \
                    * self.criterion_weights['PerformanceLoss']

        # Add performance loss to total loss
        self.loss = self.loss_perf


        if 'PSCLoss' in self.criterions.keys() and epoch > self.training_opt['pretrain']:
            discriminative, balanced = self.training_opt['discriminative_feature_space'], self.training_opt['balanced_feature_space']
            
            self.probs = F.softmax(self.logits.detach(), dim=1) # only based on classifier, without prototypes

            self.loss_psc = self.criterions['PSCLoss'](self.feat_mlp, labels, self.prototypes_mlp, self.probs, self.sample_per_class, discriminative, balanced)
            self.loss_psc = self.loss_psc * self.criterion_weights['PSCLoss']
            self.loss += self.loss_psc
        else:
            self.loss_psc = None

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat

    def train(self):

        # When training the network
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):

            for model in self.networks.values():
                model.train()
                
            torch.cuda.empty_cache()
            
            # Iterate over dataset
            for step, (inputs, labels, _) in enumerate(self.data['train']):

                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                        
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels, 
                                       centroids=self.memory['centroids'],
                                       phase='train')
                    self.batch_loss(labels)
                    self.batch_backward()

                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:

                        minibatch_loss_feat = self.loss_feat.item() \
                            if 'FeatureLoss' in self.criterions.keys() else None
                        minibatch_loss_perf = self.loss_perf.item()
                        _, preds = torch.max(self.logits, 1)
                        minibatch_acc = mic_acc_cal(preds, labels)

                        print_str = ['Epoch: [%d/%d]' 
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d' 
                                     % (step),
                                     'Minibatch_loss_feature: %.3f' 
                                     % (minibatch_loss_feat) if minibatch_loss_feat else '',
                                     'Minibatch_loss_performance: %.3f' 
                                     % (minibatch_loss_perf),
                                     'Minibatch_accuracy_micro: %.3f'
                                      % (minibatch_acc)]
                        print_write(print_str, self.log_file)

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            self.model_optimizer_scheduler.step()
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()

            # After every epoch, validation
            self.eval(phase='val')

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = copy.deepcopy(epoch)
                best_acc = copy.deepcopy(self.eval_acc_mic_top1)
                best_centroids = copy.deepcopy(self.centroids)
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        print()
        print('Training Complete.')

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)
                
        print('Done')


    def schedule_loss_weight(self, epoch, end_epoch):
        if 'PerformanceLoss' in self.criterions and 'PSCLoss' in self.criterions:
            alpha = (epoch / end_epoch)**2
            self.criterion_weights['PerformanceLoss'] = alpha
            self.criterion_weights['PSCLoss'] = 1 - alpha
            print('Losses\' weights are scheduled:')
            print('PerformanceLoss weight: {}'.format(alpha))
            print('PSCLoss weight: {}'.format(1-alpha))


    def train_for_CoMix(self):

        # When training the network
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):

            if self.training_opt['schedule_loss_weight']:
                self.schedule_loss_weight(epoch, end_epoch)

            if self.training_opt['pretrain']+1 == epoch:   # re-initiate prototypes
                if self.training_opt['reinit'] == 0:
                    self.prototypes = self.reinit_prototypes(mode=0)    # avg
                elif self.training_opt['reinit'] == 1:
                    self.prototypes = self.reinit_prototypes(mode=1)    # replace
                elif self.training_opt['reinit'] == 2:
                    self.prototypes, _ = self.init_prototypes()         # init

            for model in self.networks.values():
                model.train()
                
            torch.cuda.empty_cache()
            
            bar = tqdm(total=self.epoch_steps+1)
            # Iterate over dataset
            for step, (inputs, aug1, aug2, aug3, labels, _) in enumerate(self.data['train']):

                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break
                
                # if step == 20:
                #     break

                num_aug_to_use = len(self.training_opt['which_aug_to_use'])
                if 1 in self.training_opt['which_aug_to_use']:
                    inputs = torch.cat((inputs, aug1), dim=0)
                if 2 in self.training_opt['which_aug_to_use']:
                    inputs = torch.cat((inputs, aug2), dim=0)
                if 3 in self.training_opt['which_aug_to_use']:
                    inputs = torch.cat((inputs, aug3), dim=0)
                # inputs = torch.cat((inputs, aug1, aug2, aug3), dim=0)
                # print(labels.shape)
                labels = labels.repeat(num_aug_to_use+1)

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                        
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward_for_CoMix(inputs, labels, phase='train', epoch=epoch)
                    self.batch_loss_for_CoMix(labels, epoch)
                    self.batch_backward()


                    # update prototypes after each iteration
                    if self.memory['prototypes']:
                        if self.memory['update'] == 0:
                            self.update_prototypes(labels)
                        elif self.memory['update'] == 1:
                            self.update_prototypes_new(labels)


                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:

                        # minibatch_loss_feat = self.loss_feat.item() \
                        #     if 'FeatureLoss' in self.criterions.keys() else None
                        minibatch_loss_psc = self.loss_psc.item() \
                            if ('PSCLoss' in self.criterions.keys() and self.loss_psc) else None
                        minibatch_loss_perf = self.loss_perf.item()
                        _, preds = torch.max(self.logits, 1)
                        minibatch_acc = mic_acc_cal(preds, labels)

                        print_str = ['Epoch: [%d/%d]' 
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d' 
                                     % (step),
                                     'Minibatch_loss_psc: %.3f' 
                                     % (minibatch_loss_psc) if minibatch_loss_psc else '',
                                     'Minibatch_loss_performance: %.3f' 
                                     % (minibatch_loss_perf),
                                     'Minibatch_accuracy_micro: %.3f'
                                      % (minibatch_acc)]
                        print_write(print_str, self.log_file)
                bar.update(1)

            # self.prototypes = self.init_prototypes()

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            self.model_optimizer_scheduler.step()
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()

            # After every epoch, validation
            # self.eval_for_CoMix(phase='val')
            if self.training_opt['eval_with_prototypes'] == 2:
                self.eval(phase='val')
            else:
                self.eval(phase='val')                  # if eval_with_prototypes works
                self.eval_with_prototypes(phase='val')

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = copy.deepcopy(epoch)
                best_acc = copy.deepcopy(self.eval_acc_mic_top1)
                best_prototypes = copy.deepcopy(self.prototypes)
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        print()
        print('Training Complete.')

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model_for_CoMix(epoch, best_epoch, best_model_weights, best_acc, prototypes=best_prototypes)
                
        print('Done')

    

    def update_prototypes(self, labels):
        with torch.set_grad_enabled(False):
            _, pred = torch.max(self.logits, dim=1)  # (batch_size)
            mask = (pred.view(-1) == labels.view(-1))
            cls_num = [[0]*self.memory['prototypes_num'] for _ in range(self.training_opt['num_classes'])]
            prototypes_for_update = torch.zeros(self.training_opt['num_classes'], self.memory['prototypes_num'], self.training_opt['feature_dim']).to(self.device)
            for i, feature in enumerate(self.features):
                if not mask[i]:
                    continue
                _, which_to_update = self.prototypes[pred[i]].mm(feature.view(-1, 1)).view(-1).max(dim=0)
                prototypes_for_update[pred[i]][which_to_update] += feature
                cls_num[pred[i]][which_to_update] += 1

            for cls_idx in range(self.training_opt['num_classes']):
                for pro_idx in range(self.memory['prototypes_num']):
                    if cls_num[cls_idx][pro_idx]:
                        self.prototypes[cls_idx][pro_idx] = self.prototypes[cls_idx][pro_idx] * self.memory['ema'] + prototypes_for_update[cls_idx][pro_idx] / cls_num[cls_idx][pro_idx] * (1 - self.memory['ema'])

            print('{}/{} samples are used to update prototypes!'.format(sum([sum(line) for line in cls_num]), self.logits.shape[0]))
     
        return


    def update_prototypes_new(self, labels):
        with torch.set_grad_enabled(False):
            _, pred = torch.max(self.logits, dim=1)  # (batch_size)
            mask = (pred.view(-1) == labels.view(-1))

            cls_num = torch.zeros(self.training_opt['num_classes'], self.memory['prototypes_num'])
            prototypes_for_update = torch.zeros(self.training_opt['num_classes'], self.memory['prototypes_num'], self.training_opt['feature_dim']).to(self.device)
            
            for i, feature in enumerate(self.features):
                if not mask[i]:
                    continue
                _, which_to_update = F.normalize(self.prototypes[pred[i]], dim=1).mm(feature.view(-1, 1)).view(-1).max(dim=0)
                prototypes_for_update[pred[i]][which_to_update] += feature
                cls_num[pred[i]][which_to_update] += 1

            for cls_idx in range(self.training_opt['num_classes']):
                for pro_idx in range(self.memory['prototypes_num']):
                    if cls_num[cls_idx][pro_idx]:
                        self.prototypes[cls_idx][pro_idx] = F.normalize(self.prototypes[cls_idx][pro_idx], dim=0) * self.memory['ema'] + F.normalize(prototypes_for_update[cls_idx][pro_idx] / cls_num[cls_idx][pro_idx], dim=0) * (1 - self.memory['ema'])

            print('{}/{} samples are used to update prototypes!'.format(mask.sum(), self.logits.shape[0]))
            print('How many prototypes are updated per class:')
            print((cls_num > 0).sum(dim=1))


        return



    def eval(self, phase='val', openset=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f' 
                  % self.training_opt['open_threshold'])
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_paths = np.empty(0)

        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward_for_CoMix(inputs, labels, phase=phase)
                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))
                self.total_paths = np.concatenate((self.total_paths, paths))

        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                            self.total_labels[self.total_labels == -1])
            print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1= mic_acc_cal(preds[self.total_labels != -1],
                                            self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1 = shot_acc(preds[self.total_labels != -1],
                                     self.total_labels[self.total_labels != -1], 
                                     self.data['train'])
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s' 
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f' 
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.3f' 
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.3f' 
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f' 
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f' 
                     % (self.low_acc_top1),
                     '\n']

        if phase == 'val' or phase == 'test':
            print_write(print_str, self.log_file)
        else:
            print(*print_str)

    def eval_for_CoMix(self, phase='val', openset=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f' 
                  % self.training_opt['open_threshold'])
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_paths = np.empty(0)

        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                # self.batch_forward(inputs, labels, 
                #                    centroids=self.memory['centroids'],
                #                    phase=phase)
                self.batch_forward_for_CoMix(inputs, labels, phase=phase)
                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))
                self.total_paths = np.concatenate((self.total_paths, paths))

        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                            self.total_labels[self.total_labels == -1])
            print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1= mic_acc_cal(preds[self.total_labels != -1],
                                            self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1 = shot_acc(preds[self.total_labels != -1],
                                     self.total_labels[self.total_labels != -1], 
                                     self.data['train'])
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s' 
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f' 
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.3f' 
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.3f' 
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f' 
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f' 
                     % (self.low_acc_top1),
                     '\n']

        if phase == 'val' or phase == 'test':
            print_write(print_str, self.log_file)
        else:
            print(*print_str)


    def eval_with_prototypes(self, phase='val', openset=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f' 
                  % self.training_opt['open_threshold'])
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_paths = np.empty(0)

        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                # self.batch_forward(inputs, labels, 
                #                    centroids=self.memory['centroids'],
                #                    phase=phase)
                self.batch_forward_for_CoMix(inputs, labels, phase=phase)

                # complement with prototypes
                prototypes_for_eval = F.normalize(self.prototypes.view(-1, self.training_opt['feature_dim']), dim=1).view(self.training_opt['num_classes'], -1, self.training_opt['feature_dim'])   # (num_classes, feat_dim)
                prototypes_for_eval *= self.networks['classifier'].module.scale # to have the same magnitude of self.logits
                # complementary_logits = F.normalize(self.features, dim=1).mm(prototypes_for_eval.T)   # (batch_size, num_classes)
                complementary_logits, _ = prototypes_for_eval.matmul(F.normalize(self.features, dim=1).T).permute(2, 0, 1).max(dim=2)   # (batch_size, num_classes)
                if self.training_opt['eval_with_prototypes'] == 0:
                    self.logits = torch.where(self.logits > complementary_logits, self.logits, complementary_logits)  # choose the closer one for prediction, max
                elif self.training_opt['eval_with_prototypes'] == 1:
                    self.logits = (self.logits + complementary_logits) / 2    # average

                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))
                self.total_paths = np.concatenate((self.total_paths, paths))

        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                            self.total_labels[self.total_labels == -1])
            print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1= mic_acc_cal(preds[self.total_labels != -1],
                                            self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        self.eval_f_measure_of_LUNA = F_measure_of_LUNA(preds, self.total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        # self.eval_f_measure_of_CoMix = F_measure_of_CoMix(preds, self.total_labels, openset=openset,
        #                                 theta=self.training_opt['open_threshold'])
        self.eval_f_measure_of_CoMix = None
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1 = shot_acc(preds[self.total_labels != -1],
                                     self.total_labels[self.total_labels != -1], 
                                     self.data['train'])
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s' 
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f' 
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.3f' 
                     % (self.eval_f_measure),
                     '\n',
                     'F-measure of LUNA: %.3f' 
                     % (self.eval_f_measure_of_LUNA) if self.eval_f_measure_of_LUNA else '',
                     '\n' if self.eval_f_measure_of_LUNA else '',
                     'F-measure of CoMix: %.3f' 
                     % (self.eval_f_measure_of_CoMix) if self.eval_f_measure_of_CoMix else '',
                     '\n' if self.eval_f_measure_of_CoMix else '',
                     'Many_shot_accuracy_top1: %.3f' 
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f' 
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f' 
                     % (self.low_acc_top1),
                     '\n']

        if phase == 'val' or phase == 'test':
            print_write(print_str, self.log_file)
        else:
            print(*print_str)


            
    def centroids_cal(self, data):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                   self.training_opt['feature_dim']).cuda()

        print('Calculating centroids.')

        for model in self.networks.values():
            model.eval()

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            
            for inputs, labels, _ in tqdm(data):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]

        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids

    def load_model(self):
            
        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')
        
        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        
        for key, model in self.networks.items():

            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            # model.load_state_dict(model_state[key])
            model.load_state_dict(weights)
    
    def load_model_for_CoMix(self):
            
        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')
        
        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        self.prototypes = checkpoint['prototypes'] if 'prototypes' in checkpoint else None
        
        for key, model in self.networks.items():

            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            # model.load_state_dict(model_state[key])
            model.load_state_dict(weights)
        
    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'centroids': centroids}

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)
    
    def save_model_for_CoMix(self, epoch, best_epoch, best_model_weights, best_acc, prototypes=None):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'prototypes': prototypes}

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)
            
    def output_logits(self, openset=False):
        filename = os.path.join(self.training_opt['log_dir'], 
                                'logits_%s'%('open' if openset else 'close'))
        print("Saving total logits to: %s.npz" % filename)
        np.savez(filename, 
                 logits=self.total_logits.detach().cpu().numpy(), 
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)
