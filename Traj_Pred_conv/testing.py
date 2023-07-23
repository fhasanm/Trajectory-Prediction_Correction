from __future__ import print_function
import torch
from model import highwayNet
from utils import ngsimDataset, maskedNLL, maskedMSETest, maskedNLLTest
from torch.utils.data import DataLoader
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
args = {}
args['use_cuda'] = False
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13, 3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = False
args['train_flag'] = False

if __name__ == '__main__':

    net = highwayNet(args)
    net.load_state_dict(torch.load('trained_models/cslstm_m_7.pt'))
    net = net.to(device)
    tsSet = ngsimDataset(r'C:\Users\Hailong\Projects\ngsim_preprocessed_oldgit\TestSet.mat')
    tsDataloader = DataLoader(tsSet, batch_size=128, shuffle=True, num_workers=8, collate_fn=tsSet.collate_fn)

    for i, data in enumerate(tsDataloader):
        st_time = time.time()
        hist, nbrs, _, _, _, fut_label, _ = data

        # Initialize Variables

        hist.to(device)
        nbrs.to(device)
        # mask = mask.to(device)
        #op_mask = torch.ones(25,1,2)
        fut_label.to(device)
        masks = torch.ones([hist.shape[1], 3, 13, 64], device="cpu").bool()

        # Forward pass
        st_time = time.time()
        fut_pred = net(hist, nbrs, masks).to(device)
        time_taken = time.time() - st_time
        print(fut_pred - fut_label)

        break