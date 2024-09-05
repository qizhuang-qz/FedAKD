from torch.utils.data import DataLoader, Dataset
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def make_env(dataroot, n_labels, n_context, n_env, context_shuffle,transform):
    image_env = []
    label_env = []
    context_env = []
    if n_env > n_context:
        print('Error: There are more environments than contexts.')
    env_cont = []
    if context_shuffle:
        import random
        for l in range(n_labels):
            env_cont.append(random.shuffle(range(n_context)))
    else:
        for l in range(n_labels):
            env_cont.append(range(n_context))

    for env_idx in range(n_env):
        image_env.append([])
        label_env.append([])
        context_env.append([])
    label_names = os.listdir(dataroot)
    for label_idx in range(n_labels):
        context_names = os.listdir(dataroot + label_names[label_idx] + '/')
        for env_idx in range(n_env):
            context_idx = env_cont[label_idx][env_idx]
            path = dataroot + label_names[label_idx] + '/' + context_names[context_idx] + '/'
            image_names = os.listdir(path)
            for img in image_names:
                try:
                    temp=transform(Image.open(os.path.join(path, img)).convert('RGB'))
                    image_env[env_idx].append(temp)
                except IOError:
                    print('Warning: Broken file at ' + os.path.join(path, img))
                label_env[env_idx].append(label_idx)
                context_env[env_idx].append(context_idx)

    return image_env, label_env,context_env


def make_test(dataroot, n_labels, n_context, n_env, transform):
    all_image = []
    all_label = []
    all_context = []
    if n_env > n_context:
        print('Error: There are more environments than contexts.')
    label_names = os.listdir(dataroot)
    for label_idx in range(n_labels):
        context_names = os.listdir(dataroot + label_names[label_idx] + '/')
        for context_idx in range(n_env, n_context):
            path = dataroot + label_names[label_idx] + '/' + context_names[context_idx] + '/'
            image_names = os.listdir(path)
            for img in image_names:
                try:
                    temp=transform(Image.open(os.path.join(path, img)).convert('RGB'))
                    all_image.append(temp)
                except IOError:
                    print('Warning: Broken file at ' + os.path.join(path, img))
                all_label.append(label_idx)
                all_context.append(context_idx)

    return all_image, all_label, all_context


class NICO_dataset(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label, all_context, transform=None, require_context=False, soft_split=None,
                 label2train=None):
        super(NICO_dataset, self).__init__()
        self.all_data = all_data
        self.all_label = all_label
        self.all_context = all_context
        self.transform = transform
        self.require_context = require_context

        if label2train is None:
            label_set = list(set(self.all_label))
            label_set.sort()
            self.label2train = {label_set[i]: i for i in range(len(label_set))}
        else:
            self.label2train = label2train

        if soft_split is not None:
            self.soft_split = soft_split
        else:
            self.soft_split = None

    def __getitem__(self, item):
        img = self.all_data[item]
        img = self.transform(img)

        label = self.label2train[self.all_label[item]]
        context = self.all_context[item]

        if self.require_context:
            return img, label, context

        if self.soft_split is not None:
            return img, label, item

        return img, label

    def __len__(self):
        return len(self.all_data)


class NICO_dataset_env(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label, all_context, env_idx, transform=None, label2train=None):
        super(NICO_dataset_env, self).__init__()
        self.all_data = all_data
        self.all_label = all_label
        self.all_context = all_context
        self.transform = transform
        self.env_idx = env_idx

        if label2train is None:
            label_set = list(set(self.all_label))
            label_set.sort()
            self.label2train = {label_set[i]: i for i in range(len(label_set))}
        else:
            self.label2train = label2train

    def __getitem__(self, item):
        img = self.all_data[item]
        #img = self.transform(img)

        label = self.label2train[self.all_label[item]]
        context = self.all_context[item]

        return img, label, self.env_idx

    def __len__(self):
        return len(self.all_data)


class init_training_dataloader():
    def __init__(self, path, n_context, n_labels=10):
        super(init_training_dataloader, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                                std=[0.21851876, 0.2175944, 0.22552039])
        ])
        self.path = path
        self.n_labels = n_labels
        self.n_context = n_context
    def get_image(self,n_env, context_shuffle=False, batch_size=64, num_workers=1, shuffle=True,
                           pre_split=None):
        image_, label_, context_ = make_env(self.path, self.n_labels, self.n_context, n_env, context_shuffle,
                                            self.transform)

        return image_,label_
    def get_env_dataloader(self, n_env, context_shuffle=False, batch_size=64, num_workers=1, shuffle=True,
                           pre_split=None):
        image_, label_,context_= make_env(self.path, self.n_labels, self.n_context, n_env, context_shuffle,self.transform)
        training_dataset = []
        training_loader = []
        #print(self.transform(image_[0][0]).size())
        #print(self.transform(image_[0][1]).size())
        for env_idx in range(n_env):
            training_dataset.append(NICO_dataset_env(image_[env_idx], label_[env_idx], context_[env_idx], env_idx,
                                                     transform=self.transform))

        for i in range(n_env):
            training_loader.append(
            DataLoader(training_dataset[i], shuffle=shuffle, num_workers=num_workers,
                       batch_size=batch_size))
        return training_loader


def get_test_dataloader(path, n_labels, n_context, n_env, batch_size=64, num_workers=2, shuffle=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52418953, 0.5233741, 0.44896784],
                             std=[0.21851876, 0.2175944, 0.22552039])
    ])
    image, label, context = make_test(path, n_labels, n_context, n_env,transform_test)
    #testing_dataset = NICO_dataset(image, label, context, transform_test, require_context=False)
    #testing_loader = DataLoader(
        #testing_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    #return testing_loader
    return image,label


class CycleConcatDataset(Dataset):
    '''Dataset wrapping multiple train datasets
    Parameters
    ----------
    *datasets : sequence of torch.utils.data.Dataset
        Datasets to be concatenated and cycled
    '''

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        result = []
        for dataset in self.datasets:
            cycled_i = i % len(dataset)
            result.append(dataset[cycled_i])

        return tuple(result)

    def __len__(self):
        return max(len(d) for d in self.datasets)