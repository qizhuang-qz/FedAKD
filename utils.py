from torchvision.datasets import MNIST
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from model import *
import datasets
from datasets import CIFAR10_truncated, CIFAR100_truncated, ImageFolder_custom
import torch.autograd as autograd

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)







ALGORITHMS = [
    'ERM',
    'FedAvg',
    'FedIIR'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = Featurizer(input_shape, self.hparams)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


# FedAvg
class FedAvg(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams, args=None):
        super(FedAvg, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = args.device

    def create_client(self, ):
        self.featurizer_client = copy.deepcopy(self.featurizer)
        self.classifier_client = copy.deepcopy(self.classifier)
        self.optimizer_client = torch.optim.SGD(
            list(self.featurizer_client.parameters()) + list(self.classifier_client.parameters()),
            lr=self.hparams["lr"],
            momentum=0.9,
            weight_decay=self.hparams['weight_decay']
        )

    def aggregation_client(self, model_client):

        def aggregation(weights):

            weights_avg = copy.deepcopy(weights[0])
            for k in weights_avg.keys():
                for i in range(1, num_client):
                    weights_avg[k] += weights[i][k]
                weights_avg[k] = torch.div(weights_avg[k], num_client)

            return weights_avg

        num_model = len(model_client[0])  # number of model
        num_client = len(model_client)  # the number of client
        weights_avg = []
        for i in range(num_model):
            weights = []
            for _, total_weights in enumerate(model_client):
                weights.append(total_weights[i])
            weights_avg.append(aggregation(weights))

        self.featurizer.load_state_dict(weights_avg[0])
        self.classifier.load_state_dict(weights_avg[1])

    def update(self, sampled_clients, steps):

        model_client = []
        for _, client_data in sampled_clients:
            client_model_dict = {}
            self.create_client()

            for step in range(steps):
                for x, y in client_data:
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer_client.zero_grad()
                    feature = self.featurizer_client(x)
                    logits = self.classifier_client(feature)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    self.optimizer_client.step()

            client_model_dict['F'] = self.featurizer_client.state_dict()
            client_model_dict['C'] = self.classifier_client.state_dict()
            model_client.append([client_model_dict['F'], client_model_dict['C']])
        self.aggregation_client(model_client)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


# FedIIR
class FedIIR(FedAvg):
    def __init__(self, input_shape, num_classes, num_domains, hparams, args=None):
        super(FedIIR, self).__init__(input_shape, num_classes, num_domains, hparams, args)

        self.global_epoch = 0
        params = list(self.classifier.parameters())
        self.grad_mean = tuple(torch.zeros_like(p).to(self.device) for p in params)

    def mean_grad(self, sampled_clients):

        total_batch = 0
        grad_sum = tuple(torch.zeros_like(g).to(self.device) for g in self.grad_mean)
        for _, client_data in sampled_clients:

            for x, y in client_data:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.featurizer(x)
                logits = self.classifier(feature)
                loss = F.cross_entropy(logits, y)
                grad_batch = autograd.grad(loss, self.classifier.parameters(), create_graph=False)

                grad_sum = tuple(g1 + g2 for g1, g2 in zip(grad_sum, grad_batch))
                total_batch += 1

        grad_mean_new = tuple(grad / total_batch for grad in grad_sum)
        return tuple(self.hparams['ema'] * g1 + (1 - self.hparams['ema']) * g2
                     for g1, g2 in zip(self.grad_mean, grad_mean_new))

    def update(self, sampled_clients, steps):

        penalty_weight = self.hparams['penalty']
        self.grad_mean = self.mean_grad(sampled_clients)
        model_client = []
        for _, client_data in sampled_clients:
            client_model_dict = {}
            self.create_client()

            for step in range(steps):
                for x, y in client_data:
                    x, y = x.to(self.device), y.to(self.device)

                    self.optimizer_client.zero_grad()

                    feature = self.featurizer_client(x)
                    logits = self.classifier_client(feature)

                    loss_erm = F.cross_entropy(logits, y)
                    grad_client = autograd.grad(loss_erm, self.classifier_client.parameters(), create_graph=True)
                    # compute trace penalty
                    penalty_value = 0
                    for g_client, g_mean in zip(grad_client, self.grad_mean):
                        penalty_value += (g_client - g_mean).pow(2).sum()
                    loss = loss_erm + penalty_weight * penalty_value

                    loss.backward()
                    self.optimizer_client.step()

            client_model_dict['F'] = self.featurizer_client.state_dict()
            client_model_dict['C'] = self.classifier_client.state_dict()
            model_client.append([client_model_dict['F'], client_model_dict['C']])
        self.aggregation_client(model_client)

        self.global_epoch += 1

        return {'loss': loss_erm.item(), 'penalty': penalty_value.item()}

def transform_list_to_tensor(model_params_list):
    try:
        for k in model_params_list.keys():
            model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
        return model_params_list
    except Exception as e:
        logging.error()


def transform_tensor_to_list(model_params):
    try:
        for k in model_params.keys():
            model_params[k] = model_params[k].detach().numpy().tolist()
        return model_params
    except Exception as e:
        logging.error()

def sort_rows(matrix, num_rows):

    sorted_matrix = np.copy(matrix)


    for i in range(num_rows):
        sorted_matrix[i, :] = np.sort(matrix[i, :])

    return sorted_matrix

def slice_wasserstein(pre_img, pre_word):
    pre_img = pre_img.detach().cpu().numpy()
    pre_word = pre_word.detach().cpu().numpy()
    dim = pre_img.shape
    # 81, 128
    proj = np.random.normal(size=(dim[1], 128))
    proj *= 1 / (np.sqrt(np.sum(np.square(proj), axis=0, keepdims=True)))
    # bs, 81 @ 81, 128 = bs, 128
    p1 = np.matmul(pre_img, proj)
    p2 = np.matmul(pre_word, proj)
    p1 = sort_rows(p1, dim[0])
    p2 = sort_rows(p2, dim[0])
    wdist = np.mean(np.square(p1 - p2))
    return torch.mean(torch.tensor(wdist))


def slice_wasserstein_torch(pre_img, pre_word):
    s = pre_img.shape
    proj = torch.randn(s[1], 128).cuda()
    proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
    p1 = torch.matmul(pre_img, proj)
    p2 = torch.matmul(pre_word, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1 - p2
    wdist = torch.mean(torch.mul(dist, dist))
    return wdist



def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = datasets.MNIST(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = datasets.MNIST(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.targets
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.targets

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'./train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'./val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.5):
    np.random.seed(1000)
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset=="mnist":
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    _, _, out = model(x)
                    if len(target)==1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                _,_,out = model(x)
                #out=model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss

def compute_loss(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return avg_loss


class MNIST_truncated(MNIST):
    def __init__(self, *args, dataidxs=None, **kwargs):
        super().__init__(*args, **kwargs)

        if dataidxs is not None:
            self.data = self.data[dataidxs]
            self.targets = self.targets[dataidxs]


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.cuda()
    return model

def js_divergence(p,q,device):
    kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)
#     ipdb.set_trace()
    half=torch.div(p+q,2)
    s1=kl_loss(F.log_softmax(p, dim=1), F.softmax(half.detach(), dim=1))
    s2=kl_loss(F.log_softmax(q, dim=1), F.softmax(half.detach(), dim=1))
#     ipdb.set_trace()
    return torch.div(s1+s2,2)

def kl_divergence(p, q,device):
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean').to(device)
    log_p = F.log_softmax(p, dim=1)
    q = F.softmax(q, dim=1)
    kl_div = kl_loss(log_p, q)
    return kl_div
from torch.utils.data import Dataset, DataLoader

class NICO_dataset(torch.utils.data.Dataset):
    def __init__(self, all_data, all_label):
        super(NICO_dataset, self).__init__()
        self.all_data = all_data
        self.all_label = all_label


    def __getitem__(self, item):
        img = self.all_data[item]


        label = self.all_label[item]


        return img, label

    def __len__(self):
        return len(self.all_data)

class color_mnist_dataloader(Dataset):

        def __init__(self, features, labels, transform=None):
            self.features = features
            self.labels = labels


        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            feature = self.features[idx]
            label = self.labels[idx]


            return feature, label





def get_color_mnist_dataloader(id,beta):

    loaded_data=torch.load('fmnist_{}_train/{}_color_data_{}.pt'.format(beta,id,beta))
    train_labels=torch.load('fmnist_{}_train/{}_label_{}.pt'.format(beta,id,beta))
    test_data=torch.load('fmnist_{}_test/test_color_data_{}.pt'.format(beta,beta))
    test_labels=torch.load('fmnist_{}_test/test_label_{}.pt'.format(beta,beta))
    #print(loaded_data.size())
    dataset=color_mnist_dataloader(loaded_data,train_labels)


    dataset_test=color_mnist_dataloader(test_data,test_labels)
    train_dl=DataLoader(dataset,batch_size=64,shuffle=True)
    #test_dl=DataLoader(dataset_test,batch_size=64,shuffle=False)
    test_dl = DataLoader(dataset_test, batch_size=1, shuffle=False)

    return train_dl,test_dl

def get_NICO_dataloader(id):
    seed = 1000
    torch.manual_seed(seed)
    random.seed(seed)

    loaded_data = torch.load('NICO/nico_client{}.pt'.format(id+1))
    loaded_label=torch.load('NICO/nico_client{}_label.pt'.format(id+1))
    test_data=torch.load('NICO_test/nico_test.pt')
    test_labels=torch.load('NICO_test/nico_test_label.pt')
    dataset=NICO_dataset(loaded_data,loaded_label)
    dataset_test=NICO_dataset(test_data,test_labels)
    train_dl=DataLoader(dataset,batch_size=64,shuffle=True)
    test_dl=DataLoader(dataset_test,batch_size=64,shuffle=False)

    return train_dl,test_dl


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0):


    if dataset == 'mnist':
        dl_obj = MNIST_truncated
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_ds = dl_obj(datadir, train=True, transform=transform, download=True, dataidxs=dataidxs)
        test_ds = dl_obj(datadir, train=False, transform=transform, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset in ('cifar10', 'cifar100'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                #transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])



        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)


    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir+'./train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'./val/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)





    return train_dl, test_dl, train_ds, test_ds


import cv2
import numpy as np
import torch
import torch.nn.functional as F



def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    heatmap = F.interpolate(heatmap, size=(img.size(2), img.size(3)), mode='bilinear', align_corners=False)
    heatmap = heatmap.to(img.device)
    img = img.to(heatmap.device)
    result = heatmap + img
    result = result.div(result.max()).squeeze()

    return heatmap, result


def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_vgg_layer(arch, target_layer_name):
    """Find vgg layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_42'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_alexnet_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_squeezenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features_12'
            target_layer_name = 'features_12_expand3x3'
            target_layer_name = 'features_12_expand3x3_activation'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2] + '_' + hierarchy[3]]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)