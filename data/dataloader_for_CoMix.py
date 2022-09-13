from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image

from randaugment import rand_augment_transform, GaussianBlur

# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([                                             # val and test have the same transform
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}



# augmentations for training PSC Loss

randaug_n, randaug_m = 2, 10    # as in BCL
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
rgb_mean = (0.485, 0.456, 0.406)
ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
augmentation_randncls = [
    transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
    ], p=1.0),
    rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params), # only transform more than train
    transforms.ToTensor(),
    normalize,
]
augmentation_randnclsstack = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),  # transform more than train
    rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params), # transform more than train
    transforms.ToTensor(),
    normalize,
]
augmentation_sim = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),  # only transform more than train
    transforms.ToTensor(),
    normalize
]


# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None, transform_psc=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.transform_psc = transform_psc
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample1 = self.transform(sample)

        if not self.transform_psc:  # eval or test
            return sample1, label, path

        # aug for PSC Loss
        aug1 = self.transform_psc[0](sample)
        aug2 = self.transform_psc[1](sample)
        aug3 = self.transform_psc[2](sample)
        
        return sample1, aug1, aug2, aug3, label, path



# Load datasets
def load_data(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True, aug_for_psc=0):
    
    txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print('Loading data from %s' % (txt))

    if phase not in ['train', 'val']:
        transform = data_transforms['test']
    else:
        transform = data_transforms[phase]


    # augmentations for training PSC Loss
    if aug_for_psc == 0:    # 'sim-sim'
        transform_psc = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                            transforms.Compose(augmentation_sim), ]
    elif aug_for_psc == 1:    # 'sim-rand'
        transform_psc = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                            transforms.Compose(augmentation_sim), ]
    elif aug_for_psc == 2:    #'randstack-randstack'
        transform_psc = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                            transforms.Compose(augmentation_randnclsstack), ]
    elif aug_for_psc == 3:
        transform_psc = [data_transforms[phase], data_transforms[phase], data_transforms[phase]]
    else:
        raise NotImplementedError("This augmentations strategy is not available for contrastive learning branch!")

    if phase in ['val', 'test']:
        transform_psc = None




    print('Use data transformation:', transform)
    print('Use data transformation for psc:', transform_psc)

    set_ = LT_Dataset(data_root, txt, transform, transform_psc)

    if phase == 'test' and test_open:
        open_txt = './data/%s/%s_open.txt'%(dataset, dataset)
        print('Testing with opensets from %s'%(open_txt))
        open_set_ = LT_Dataset('./data/%s/%s_open'%(dataset, dataset), open_txt, transform)
        set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler.')
        print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, sampler_dic['num_samples_cls']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
        
    
    
