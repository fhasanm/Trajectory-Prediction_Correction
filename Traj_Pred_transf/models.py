from typing_extensions import OrderedDict
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from layers.graph import Graph
from layers.attention import PositionwiseFeedForward, MultiHeadAttentionNew, EncoderLayer
from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import EncoderRNN, Seq2Seq, TransformerSeq2Seq, DecoderRNN



class SpatialTransformerRNN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, in_size, out_size, d_model=128, d_inner=512,
            n_layers=2, n_head=4, d_k=32, d_v=32, dropout=0.1, n_position=16, spatial_interact=True, **kwargs):

        super().__init__()

        self.d_model = d_model
        transformer_encoder = Encoder
        if spatial_interact:
            transformer_encoder = SpatialEncoder
        self.transformer_encoder = transformer_encoder(
            in_size=in_size, n_position=n_position,
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, scale_emb=False)
        self.out_size = out_size  # 2
        self.seq2seq_type = kwargs.get('seq2seq_type', 'gru')
        self.interact_in_decoding = kwargs.get('interact_in_decoding', False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.rnn_encoder = EncoderRNN(self.seq2seq_type, input_size=d_model, hidden_size=out_size, num_layers=2)
        self.rnn_decoder = DecoderRNN(self.seq2seq_type, hidden_size=out_size, output_size=out_size, num_layers=2,
                                      dropout=0.5)
        if self.interact_in_decoding:
            d_model = 2 * 60
            d_k = d_v = d_model // n_head
            self.layer_norm = nn.LayerNorm(60)
            self.attention_interact = MultiHeadAttentionNew(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v,
                                                            dropout=dropout)
        # self.decoder = DecoderRNN(type='gru', num_layers=n_layers, output_size=out_size, hidden_size=d_model//n_layers, dropout=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, pra_x, pra_A, pra_pred_length, input_mask=None, teacher_forcing_ratio=0.0,
                pra_teacher_location=None, is_train=True, **kwargs):

        batch_size, in_size, enc_seq_len, num_object = pra_x.size()


        pra_x = rearrange(pra_x, 'bs is sl no -> (bs no) sl is')

        if teacher_forcing_ratio > 0 and type(pra_teacher_location) is not type(None):
            pra_teacher_location = rearrange(pra_teacher_location, 'bs is sl no -> (bs no) sl is')
            assert pra_pred_length == pra_teacher_location.size(-2)

        velocity_locations = torch.zeros((batch_size * num_object, pra_pred_length, self.out_size), device=pra_x.device)

        src_mask = None  # get_pad_mask(src_seq, self.src_pad_idx)
        spatial_mask = None

        if type(pra_A) is not type(None):
            spatial_mask = rearrange(pra_A, 'b l m n -> (b l) m n').bool() # (bs * sl, no, no)

        dec_spatial_mask = None

        if type(input_mask) is not type(None):

            # input_mask: (bs, 1, sl, no)
            src_mask = rearrange(input_mask, 'bs is sl no -> (bs no) sl is')  # (bs * no, sl, 1)
            subsequent_mask = get_subsequent_mask(src_mask.squeeze(-1))
            src_mask = torch.einsum('bsi,bxi->bsx', src_mask, src_mask)
            src_mask = (subsequent_mask * src_mask).bool()


            if self.interact_in_decoding:
                # we only allow the message passing among the observed vehicles which at least have one valid position information (local_x, local_y).
                dec_spatial_mask = rearrange(input_mask, 'bs is sl no -> bs no (sl is)')  # (bs, no, sl)
                dec_spatial_mask = dec_spatial_mask.sum(axis=-1, keepdim=True)  # (bs, no, 1)
                dec_spatial_mask = torch.einsum('boi,bui->bou', dec_spatial_mask,
                                                dec_spatial_mask).bool()  # (batch_size, num_object, num_object)
            #print(batch_size)

        transformer_output, *_ = self.transformer_encoder(pra_x, src_mask, spatial_mask=spatial_mask,
                                                          batch_size=batch_size)  # enc_output: (bs * num_object, history_frames, hidden_size)
        last_position_velocity = pra_x[:, -1:, :2]

        encoded_output, hidden = self.rnn_encoder(transformer_output)

        dec_input = last_position_velocity  # the last observed position, (bs * num_object, 1, in_size)

        for t in range(pra_pred_length):
            now_out, hidden = self.rnn_decoder(dec_input, hidden)
            # now_out += last_position_velocity
            velocity_locations[:, t:t + 1] = now_out
            teacher_force = np.random.random() < teacher_forcing_ratio
            last_position_velocity = (pra_teacher_location[:, t:t + 1] if (type(pra_teacher_location) is not type(
                None)) and teacher_force else now_out)
            dec_input = last_position_velocity
            if self.interact_in_decoding:
                hidden = self.message_passing(hidden, mask=dec_spatial_mask, batch_size=batch_size)

        outputs = rearrange(velocity_locations, '(bs no) sl hs -> bs hs sl no', bs=batch_size)


        return outputs, None

    def message_passing(self, hidden, batch_size, mask=None):
        # hidden: (num_layers, batch_size * num_object, hidden_size)
        shaped_hidden = rearrange(hidden, 'nl (b o) hs -> b o (nl hs)',
                                  b=batch_size)  # (batch_size, num_object, num_layers * hidden_size)
        interacted_hidden, _ = self.attention_interact(shaped_hidden, shaped_hidden, shaped_hidden, mask=mask)
        interacted_hidden = rearrange(interacted_hidden, 'b o (nl hs) -> nl (b o) hs',
                                      nl=self.rnn_decoder.num_layers).contiguous()
        interacted_hidden = self.dropout(interacted_hidden)
        hidden = self.layer_norm(hidden + interacted_hidden)
        return hidden


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    len_s = seq.size(-1)
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class SpatialEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(SpatialEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttentionNew(n_head, d_model, d_k, d_v, dropout=dropout)
        self.spatial_attn = MultiHeadAttentionNew(n_head, d_model, d_k, d_v, dropout=dropout)
        # self.temporal_attn = MultiHeadAttentionNew(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, batch_size, slf_attn_mask=None, spatial_attn_mask=None):
        self_attn_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        # bs_v, seq_len, h_size = enc_output.size()
        # (batch_size * num_object, seq_len, hidden_size) -> (batch_size, num_object, seq_len, hidden_size)
        # -> (batch_size, seq_len, num_object, hidden_size) -> (batch_size * seq_len, num_object, hidden_size)
        spatial_attn_input = rearrange(self_attn_output, '(bs no) sl hs -> (bs sl) no hs', bs=batch_size)

        spatial_attn_output, enc_spaital_attn = self.spatial_attn(
            spatial_attn_input, spatial_attn_input, spatial_attn_input, mask=spatial_attn_mask
        )
        spatial_attn_output = rearrange(spatial_attn_output, '(bs sl) no hs -> (bs no) sl hs', bs=batch_size)

        enc_output = self.pos_ffn(spatial_attn_output)

        return enc_output, enc_slf_attn, enc_spaital_attn





class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class AbsolutePositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(AbsolutePositionalEncoding, self).__init__()
        self.postional_embeds = nn.Embedding(n_position, d_hid)

    def forward(self, x):
        position = torch.LongTensor([i for i in range(x.size(1))]).unsqueeze(0).to(x.device)
        position_embeds = self.postional_embeds(position)
        return x + position_embeds


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, in_size, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Linear(in_size, d_model, bias=False)
        self.position_enc = AbsolutePositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, batch_size, return_attns=False, **kwargs):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        nv, seq_len, hidden_size = enc_output.size()
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class SpatialEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, in_size, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Linear(in_size, d_model, bias=False)
        self.position_enc = AbsolutePositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            SpatialEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, spatial_mask, batch_size, return_attns=False, **kwargs):

        enc_slf_attn_list, enc_spatial_list = [], []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        nv, seq_len, hidden_size = enc_output.size()
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn, enc_spatial_attn = enc_layer(
                enc_output, batch_size=batch_size,
                slf_attn_mask=src_mask, spatial_attn_mask=spatial_mask
            )
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            enc_spatial_list += [enc_spatial_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list, enc_spatial_attn

        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, in_size, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Linear(in_size, d_model, bias=False)
        self.position_enc = AbsolutePositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            SpatialDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, dec_spatial_mask, batch_size, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list, dec_spatial_attn_list = [], [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)

        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn, dec_spatial_attn = dec_layer(
                dec_output, enc_output, batch_size=batch_size, slf_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask, dec_spatial_attn_mask=dec_spatial_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
            dec_spatial_attn_list += [dec_spatial_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list, dec_spatial_attn_list
        return dec_output,




if __name__ == '__main__':
    from tqdm import tqdm

    model = SpatialTransformer(in_size=4, out_size=2)
    # print(model)
    bs = 32
    for i in tqdm(range(20)):
        data = torch.randn((bs, 4, 16, 22))
        pra_teacher_location = torch.rand((bs, 4, 25, 22))
        outputs, _ = model(data, pra_pred_length=25, is_train=True, teacher_forcing_ratio=0.0,
                           pra_teacher_location=pra_teacher_location)
        # print(outputs.size())