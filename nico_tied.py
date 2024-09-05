import torch
from torch.utils.data import Dataset, DataLoader
import random
class NICO_dataset(torch.utils.data.Dataset):
    def __init__(self, all_data, data, all_label):
        super(NICO_dataset, self).__init__()
        self.all_data = all_data
        self.data=data
        self.all_label = all_label


    def __getitem__(self, item):
        img = self.all_data[item]
        img_frame=self.data[item]

        label = self.all_label[item]


        return img, img_frame,label

    def __len__(self):
        return len(self.all_data)

class NICO_dataset_2(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label):
        super(NICO_dataset_2, self).__init__()
        self.all_data = all_data

        self.all_label = all_label


    def __getitem__(self, item):
        img = self.all_data[item]


        label = self.all_label[item]


        return img, label

    def __len__(self):
        return len(self.all_data)


def get_NICO_dataloader_train(id):
    seed = 1000
    torch.manual_seed(seed)
    random.seed(seed)
    #original data
    loaded_data = torch.load('NICO2/nico_client{}.pt'.format(id+1))
    loaded_label=torch.load('NICO2/nico_client{}_label.pt'.format(id+1))
    #frame data
    frame_data = torch.load('NICO_frame/nico_client{}.pt'.format(id + 1))
    frame_label = torch.load('NICO_frame/nico_client{}_label.pt'.format(id + 1))


    dataset=NICO_dataset(loaded_data,frame_data,loaded_label)

    train_dl=DataLoader(dataset,batch_size=64,shuffle=True)


    return train_dl

def get_NICO_dataloader_test():
    seed = 1000
    torch.manual_seed(seed)
    random.seed(seed)
    #labels
    test_data=torch.load('NICO_test/nico_test.pt')
    test_labels=torch.load('NICO_test/nico_test_label.pt')
    dataset_test=NICO_dataset_2(test_data,test_labels)
    test_dl=DataLoader(dataset_test,batch_size=64,shuffle=False)

    return test_dl