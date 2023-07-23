import torch
from model import highwayNet


def traj_pred(model, tracks):
    #for each vehicle

    tracks = tracks.reshape(1, tracks.shape[0], tracks.shape[1], tracks.shape[2]) # c t n -> b c t n
    predictions = torch.zeros(2, 25, tracks.shape[3])
    for i in range(tracks.shape[3]):
        # proj = tracks[:,:,-1,i].repeat(tracks.shape[3], 1)
        # assert proj.shape == tracks.shape, "proj and tracks mismatch"
        transform = tracks[:,:,-1,i].reshape(tracks.shape[0],tracks.shape[1],1,1)
        transformed_tracks = tracks - transform # transforms the coordinate frame to target vehicle at t=0

        nbrs = transformed_tracks.reshape(transformed_tracks.shape[1],transformed_tracks.shape[2], transformed_tracks.shape[3])
        nbrs = nbrs.permute(1,2,0) #c,t,n -> t,n,c
        assert nbrs.shape == (tracks.shape[2], tracks.shape[3], tracks.shape[1]), "nbrs shape error"

        hist = transformed_tracks[:,:,:,i].reshape(transformed_tracks.shape[0],transformed_tracks.shape[1], transformed_tracks.shape[2])
        hist = hist.permute(2,0,1) #b,c,t -> t, b, c
        assert hist.shape == (tracks.shape[2], tracks.shape[0], tracks.shape[1]), "hist shape error"

        masks = torch.zeros([hist.shape[1], 3, 13, 64], device="cuda" if torch.cuda.is_available() else 'cpu').bool()
        fut = model(hist, nbrs, masks)
        fut = fut[:,:,:2]

        fut = fut.reshape(1,fut.shape[0], fut.shape[1], fut.shape[2]) #t b c -> 1 t b c
        fut = fut.permute(2,3,1,0) #n t b c -> b c t n
        fut = fut + transform
        fut = fut.reshape(fut.shape[1], fut.shape[2], fut.shape[3])

        predictions[:,:,i] = fut

        return predictions



if __name__ == '__main__':

    args = {}
    args['use_cuda'] = True
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

    model = highwayNet(args)
    model.load_state_dict(torch.load('trained_models/cslstm_m_0.pt'))
    model = highwayNet(args)
    tracks = torch.randn([2, 16, 120])

    traj_pred(model, tracks)