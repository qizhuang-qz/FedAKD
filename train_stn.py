from model import *
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import copy
from gen_nico import *
from utils import *


epochs=10
#train_data=init_training_dataloader(path="C:/Users/84210/Desktop/NICO_animal/",n_context=10,n_labels=10)
#train_dataloader=train_data.get_env_dataloader(n_env=10)
#print('train_dataloader_finished')
#X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
       #args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=0.5)
#train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               #args.datadir,
                                                                               #args.batch_size,
                                                                               #32)



for id in range(10):
   #value=net_dataidx_map[id]
   #model_cell=STN(args.dataset).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
   model_cell=STN_NICO('NICO').cuda()
   train_dl_local, _ =get_NICO_dataloader(id+1)
   #train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, value)
   #train_dl_local=train_dataloader[id]
   print('开始训练')
   optimizer = optim.SGD(model_cell.parameters(), lr=0.01)


   for epoch in range(epochs):
       model_cell.train()
       train_loss=0.0
       correct=0
       total=0
       batches_pbar = tqdm(enumerate(train_dl_local), total=len(train_dl_local),desc=f'Epoch {epoch}')

       for batch_idx, (data, target) in batches_pbar:
            data, target = data.to('cuda'), target.to('cuda')
            target=target.long()
            optimizer.zero_grad()
            output,_,_ = model_cell(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _,predicted=torch.max(output.data,1)
            total+=target.size(0)
            correct+=(predicted==target).sum().item()


            avg_train_loss = train_loss / (batch_idx + 1)
            accuracy=correct/total
            batches_pbar.set_postfix({'Average Loss': avg_train_loss,'Accuracy':accuracy}, refresh=True)

   '''with torch.no_grad():
       # Get a batch of training data
       data = next(iter(train_dl_local))[0].to(device)

       input_tensor = data.cpu()
       transformed_input_tensor = model_cell.stn(data).cpu()

       in_grid = convert_image_np(
           torchvision.utils.make_grid(input_tensor))

       out_grid = convert_image_np(
           torchvision.utils.make_grid(transformed_input_tensor))

       # Plot the results side-by-side
       f, axarr = plt.subplots(1, 2)
       axarr[0].imshow(in_grid)
       axarr[0].set_title('Dataset Images')

       axarr[1].imshow(out_grid)
       axarr[1].set_title('Transformed Images')
   
   plt.ioff()
   plt.show()'''
   model_cell.eval()
   best_model_wts = copy.deepcopy(model_cell.state_dict())  # save the model state dict
   address='stn_NICO/pre_model_{}.pt'.format(id+1)
   print(address)

   torch.save(best_model_wts, address)





















