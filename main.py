import argparse
import os
import glob
import torch
import random
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertAttention

from figaro.models.vae import VqVaeModule
from figaro.models.seq2seq import Seq2SeqModule
from figaro.datasets import MidiDataset, SeqCollator
from figaro.utils import medley_iterator
from figaro.input_representation import remi2midi
from figaro.input_representation import InputRepresentation
from figaro.vocab import RemiVocab, DescriptionVocab
from figaro.constants import (
  PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, BAR_KEY, POSITION_KEY,
  TIME_SIGNATURE_KEY, INSTRUMENT_KEY, CHORD_KEY,
  NOTE_DENSITY_KEY, MEAN_PITCH_KEY, MEAN_VELOCITY_KEY, MEAN_DURATION_KEY
)

def load_old_or_new_checkpoint(model_class, checkpoint):
    # assuming transformers>=4.36.0
    pl_ckpt = torch.load(checkpoint, map_location="cpu")
    kwargs = pl_ckpt['hyper_parameters']
    if 'flavor' in kwargs:
        del kwargs['flavor']
    if 'vae_run' in kwargs:
        del kwargs['vae_run']
    model = model_class(**kwargs)
    state_dict = pl_ckpt['state_dict']
    # position_ids are no longer saved in the state_dict starting with transformers==4.31.0
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith('embeddings.position_ids')}
    try:
        # succeeds for checkpoints trained with transformers>4.13.0
        model.load_state_dict(state_dict)
    except RuntimeError:
        # work around a breaking change introduced in transformers==4.13.0, which fixed the position_embedding_type of cross-attention modules "absolute"
        config = model.transformer.decoder.bert.config
        for layer in model.transformer.decoder.bert.encoder.layer:
            layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
        model.load_state_dict(state_dict)
    model.freeze()
    model.eval()
    return model

def load_model(checkpoint, vae_checkpoint=None, device='auto'):
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae_module = None
    if vae_checkpoint:
        vae_module = load_old_or_new_checkpoint(VqVaeModule, vae_checkpoint)
        vae_module.cpu()

    model = load_old_or_new_checkpoint(Seq2SeqModule, checkpoint)
    model.to(device)

    return model, vae_module

def get_features(file_name, ctx):
    vocab = RemiVocab()
    desc_vocab = DescriptionVocab()
    rep = InputRepresentation(file_name, strict=True)
    events = rep.get_remi_events()
    description = rep.get_description()

    # Get Bar Ids
    bars = [i for i, event in enumerate(events) if f"{BAR_KEY}_" in event]
    bar_ids = torch.bincount(torch.tensor(bars, dtype=torch.int), minlength=len(events))
    bar_ids = torch.cumsum(bar_ids, dim=0)

    # Get positions
    evnts = [f"{POSITION_KEY}_0" if f"{BAR_KEY}_" in event else event for event in events]
    position_events = [event if f"{POSITION_KEY}_" in event else None for event in evnts]

    positions = [int(pos.split('_')[-1]) if pos is not None else None for pos in position_events]

    if positions[0] is None:
        positions[0] = 0
    for i in range(1, len(positions)):
        if positions[i] is None:
            positions[i] = positions[i-1]
    position_ids = torch.tensor(positions, dtype=torch.int)

    # Else
    event_ids = torch.tensor(vocab.encode(events), dtype=torch.long)
    bos = torch.tensor(vocab.encode([BOS_TOKEN]), dtype=torch.long)
    eos = torch.tensor(vocab.encode([EOS_TOKEN]), dtype=torch.long)
    zero = torch.tensor([0], dtype=torch.int)
    event_ids = torch.cat([bos, event_ids, eos])
    bar_ids = torch.cat([zero, bar_ids, zero])
    position_ids = torch.cat([zero, position_ids, zero])

    start, end = (0, len(event_ids))
    src = event_ids[start:end]
    b_ids = bar_ids[start:end]
    p_ids = position_ids[start:end]

    x = {
        'input_ids': src,
        'file': file_name,
        'bar_ids': b_ids,
        'position_ids': p_ids,
    }

    # Assume that bar_ids are in ascending order (except for EOS)
    min_bar = b_ids[0]
    desc_events = description
    desc_bars = [i for i, event in enumerate(desc_events) if f"{BAR_KEY}_" in event]
    # subtract one since first bar has id == 1
    start_idx = desc_bars[max(0, min_bar - 1)]

    desc_bar_ids = torch.zeros(len(desc_events), dtype=torch.int)
    desc_bar_ids[desc_bars] = 1
    desc_bar_ids = torch.cumsum(desc_bar_ids, dim=0)

    desc_bos = torch.tensor(desc_vocab.encode([BOS_TOKEN]), dtype=torch.int)
    desc_eos = torch.tensor(desc_vocab.encode([EOS_TOKEN]), dtype=torch.int)
    desc_ids = torch.tensor(desc_vocab.encode(desc_events), dtype=torch.int)
    if min_bar == 0:
        desc_ids = torch.cat([desc_bos, desc_ids, desc_eos])
        desc_bar_ids = torch.cat([zero, desc_bar_ids, zero])
    else:
        desc_ids = torch.cat([desc_ids, desc_eos])
        desc_bar_ids = torch.cat([desc_bar_ids, zero])

    x['description'] = desc_ids[start:]
    x['desc_bar_ids'] = desc_bar_ids[start:]
    # x = {k:(v[:ctx] if isinstance(v, torch.Tensor) else v) for k,v in x.items()}
    return x

def sample(model, batch, temperature = 1.2, max_iter = 16000, max_bars = 1):
    batch_size, seq_len = batch['input_ids'].shape[:2]
    batch_ = { key: batch[key][:, :1] for key in ['input_ids', 'bar_ids', 'position_ids'] }
    if model.description_flavor in ['description', 'both']:
        batch_['description'] = batch['description']
        batch_['desc_bar_ids'] = batch['desc_bar_ids']

    max_len = seq_len + 1024
    if max_iter > 0:
        max_len = min(max_len, 1 + max_iter)

    pad_token_id = model.vocab.to_i(PAD_TOKEN)
    eos_token_id = model.vocab.to_i(EOS_TOKEN)

    batch_size, curr_len = batch_['input_ids'].shape

    i = curr_len - 1
    x = batch_['input_ids']
    bar_ids = batch_['bar_ids']
    position_ids = batch_['position_ids']
    assert x.shape[:2] == bar_ids.shape and x.shape[:2] == position_ids.shape, f"Input, bar and position ids weren't of compatible shapes: {x.shape}, {bar_ids.shape}, {position_ids.shape}"

    z, desc_bar_ids = batch_['description'], batch_['desc_bar_ids'].to(model.device)

    is_done = torch.zeros(batch_size, dtype=torch.bool)
    encoder_hidden_states = None

    curr_bars = torch.zeros(batch_size).to(model.device).fill_(-1)
    for i in range(curr_len - 1, max_len):
        x_ = x[:, -model.context_size:].to(model.device)
        bar_ids_ = bar_ids[:, -model.context_size:].to(model.device)
        position_ids_ = position_ids[:, -model.context_size:].to(model.device)

        if model.description_flavor in ['description', 'both']:
            if model.description_flavor == 'description':
                desc = z
            else:
                desc = z['description']
            
            next_bars = bar_ids_[:, 0]
            bars_changed = not (next_bars == curr_bars).all()
            curr_bars = next_bars

            if bars_changed:
                z_ = torch.zeros(batch_size, model.context_size, dtype=torch.int)
                desc_bar_ids_ = torch.zeros(batch_size, model.context_size, dtype=torch.int)

                for j in range(batch_size):
                    curr_bar = bar_ids_[j, 0]
                    indices = torch.nonzero(desc_bar_ids[j] == curr_bar)
                    if indices.size(0) > 0:
                        idx = indices[0, 0]
                    else:
                        idx = desc.size(1) - 1

                    offset = min(model.context_size, desc.size(1) - idx)

                    z_[j, :offset] = desc[j, idx:idx+offset]
                    desc_bar_ids_[j, :offset] = desc_bar_ids[j, idx:idx+offset]

                z_, desc_bar_ids_ = z_.to(model.device), desc_bar_ids_.to(model.device)
                encoder_hidden_states = model.encode(z_, desc_bar_ids_)

        logits = model.decode(x_, bar_ids=bar_ids_, position_ids=position_ids_, encoder_hidden_states=encoder_hidden_states)

        idx = min(model.context_size - 1, i)
        logits = logits[:, idx] / temperature

        pr = torch.nn.functional.softmax(logits, dim=-1)
        pr = pr.view(-1, pr.size(-1))

        next_token_ids = torch.multinomial(pr, 1).view(-1).to(x.device)
        next_tokens = model.vocab.decode(next_token_ids)

        next_bars = torch.tensor([1 if f'{BAR_KEY}_' in token else 0 for token in next_tokens], dtype=torch.int)
        next_bar_ids = bar_ids[:, i].clone() + next_bars

        next_positions = [f"{POSITION_KEY}_0" if f'{BAR_KEY}_' in token else token for token in next_tokens]
        next_positions = [int(token.split('_')[-1]) if f'{POSITION_KEY}_' in token else None for token in next_positions]
        next_positions = [pos if next_pos is None else next_pos for pos, next_pos in zip(position_ids[:, i], next_positions)]
        next_position_ids = torch.tensor(next_positions, dtype=torch.int)

        is_done.masked_fill_((next_token_ids == eos_token_id).all(dim=-1), True)
        next_token_ids[is_done] = pad_token_id
        if max_bars > 0:
            is_done.masked_fill_(next_bar_ids >= max_bars + 1, True)

        x = torch.cat([x, next_token_ids.clone().unsqueeze(1)], dim=1)
        bar_ids = torch.cat([bar_ids, next_bar_ids.unsqueeze(1)], dim=1)
        position_ids = torch.cat([position_ids, next_position_ids.unsqueeze(1)], dim=1)

        if torch.all(is_done):
            break
        # print()

    return {
        'sequences': x,
        'bar_ids': bar_ids,
        'position_ids': position_ids
    }

model, vae_module = load_model('./checkpoints/figaro-expert.ckpt', './checkpoints/vq-vae.ckpt')
model.to('mps')

max_iter = 16000
max_bars = 1
temperature = 1.2

prompt = './lmd_full/0/0a0a2b0e4d3b7bf4c5383ba025c4683e.mid'

x = get_features(prompt, ctx=model.context_size)
batch = {k:((v[None, :] if len(v.size()) == 1 else v) if isinstance(v,torch.Tensor) else v) for k,v in x.items()}

with torch.no_grad():
    s = sample(model, batch, temperature, max_iter, max_bars)
    
    xs = batch['input_ids'].detach().cpu()
    xs_hat = s['sequences'].detach().cpu()
    events = [model.vocab.decode(x) for x in xs]
    events_hat = [model.vocab.decode(x) for x in xs_hat]

    pms_hat = []
    n_fatal = 0
    for rec_hat in events_hat:
        pm_hat = remi2midi(rec_hat)
        pms_hat.append(pm_hat)

