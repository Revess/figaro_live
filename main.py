import torch, mido, pretty_midi, time, sys
from transformers.models.bert.modeling_bert import BertAttention

from figaro.models.vae import VqVaeModule
from figaro.models.seq2seq import Seq2SeqModule
from figaro.input_representation import remi2midi
from figaro.input_representation import InputRepresentation
from figaro.vocab import RemiVocab, DescriptionVocab
from figaro.constants import (
  PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, BAR_KEY, POSITION_KEY,
)

from figaro.live_utils import load_model, get_features, sample

import threading as td
from queue import Queue
import json
from copy import deepcopy

IGNORED_CLOCK_TICKS = 256
if __name__ == "__main__":
    with open('./cfg.json', 'r') as f:
        cfg = json.load(f)
    MIDI_IN = cfg['MIDI_IN']
    MIDI_OUT = cfg['MIDI_OUT']
    PPQN = cfg['PPQN']
    METER = cfg['METER']
    MAX_BARS = cfg['MAX_BARS']
    MAX_NOTES = cfg['MAX_NOTES']
    MAX_MEASURES = cfg['MAX_MEASURES']

# Define queues
transport_in_queue = Queue()
timing_queue = Queue()
AI_in_queue = Queue()
AI_out_queue = Queue()
transport_out_user_queue = Queue()
transport_out_AI_manager_queue = Queue()

def transport():
    ignored_ticks = 0
    tick = 0
    beat = 0

    next_ai_note = None
    notes_to_play = []
    sort = False

    with mido.open_input(MIDI_IN) as inport, mido.open_output(MIDI_OUT) as outport:
        start_time = time.time()
        tick_start = time.time()
        avg_delta = 0
        for msg in inport:
            while not transport_in_queue.empty():
                msg_ = transport_in_queue.get()
                if isinstance(msg_, mido.Message): # If it is a MIDO message immidiately play it!
                    outport.send(msg_)
                elif isinstance(msg_, tuple):
                    notes_to_play.append(msg_)
                    sort = True
            if sort:
                notes_to_play.sort(key=lambda event: event[0])
                sort = False
            if msg.type == 'clock':
                if ignored_ticks < IGNORED_CLOCK_TICKS:
                    ignored_ticks += 1
                    continue  # Skip processing until we reach stable timing

                delta = time.time() - tick_start
                avg_delta += delta

                if (tick + 1) % PPQN == 0:
                    tick = 0
                    bpm = round(60 / avg_delta, 0)
                    avg_delta = 0

                    time_ = time.time() - start_time
                    if beat % METER == 0:
                        pitch = 75
                    else:
                        pitch = 56

                    if beat % METER == 0 and beat != 0:
                        beat = 0
                        # start = time.time()
                    beat += 1

                    # Sending notes
                    outport.send(mido.Message('note_on', note=pitch, velocity=100, channel=9))
                    time.sleep(0.01)
                    outport.send(mido.Message('note_off', note=pitch, velocity=100, channel=9))
                    
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=time_,
                        end=time_ + (60 / bpm)
                    )

                    transport_out_user_queue.put(note)
                    transport_out_AI_manager_queue.put(note)

                tick += 1
                tick_start = time.time()

            if len(notes_to_play) > 0 and next_ai_note is None:
                next_ai_note = notes_to_play.pop(0)

            if next_ai_note is not None:
                print(next_ai_note[0], time.time() - start_time, next_ai_note)
                if next_ai_note[0] < (time.time() - start_time):
                    m_note = mido.Message(next_ai_note[1], note=note.pitch, velocity=note.velocity)
                    outport.send(m_note)
                    next_ai_note = None

def user_input():
    start_of_play = time.time()
    active_notes = {}
    active_measures = 0
    got_impulse = False
    start_play = False

    with mido.open_input(MIDI_IN) as inport:
        for msg in inport:
            if not transport_out_user_queue.empty():
                tick = transport_out_user_queue.get()
                last_clock_time = tick.start
                if tick.pitch == 75:
                    got_impulse = True
                st_last_clock_received = time.time()
            
            if active_measures > MAX_MEASURES * 2:
                active_measures = 0
                start_play = False
                got_impulse = False
                active_notes = {}
                # start_of_play = time.time()

            if active_measures > MAX_MEASURES:
                if got_impulse:
                    active_measures += 1
                    got_impulse = False 
                if not start_play:
                    AI_in_queue.put('start_playing')
                    start_play = True
                continue      

            if got_impulse and active_measures > 0:
                active_measures += 1
                got_impulse = False         

            if msg.type == 'note_on' and msg.velocity > 0:
                transport_in_queue.put(msg)
                active_notes[msg.note] = ((time.time() - st_last_clock_received) + last_clock_time, msg.velocity)
                if got_impulse and active_measures == 0:
                    active_measures += 1
                    got_impulse = False

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0) and msg.note in active_notes.keys():
                transport_in_queue.put(msg)
                start, velocity = active_notes.pop(msg.note)
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=msg.note,
                    start=start,
                    end=(time.time() - st_last_clock_received) 
                )

                AI_in_queue.put(note)

def AI_host():
    initial_prompt = 1
    max_iter = 16000
    temperature = 0.7

    start_playing = False
    piano = pretty_midi.Instrument(program=0)
    click_track = pretty_midi.Instrument(program=9)

    model, vae_module = load_model('./checkpoints/figaro-expert.ckpt', './checkpoints/vq-vae.ckpt')
    model.to('cuda:0')

    while True:
        if not transport_out_AI_manager_queue.empty():
            note_ = transport_out_AI_manager_queue.get() # Order
            click_track.notes.append(note_) # Order

        if not AI_in_queue.empty():
            msg = AI_in_queue.get()
            if isinstance(msg, pretty_midi.Note):
                piano.notes.append(msg)
                if len(piano.notes) > MAX_NOTES:
                    note_ = piano.notes.pop(0)
                    break_point = 0
                    for step, click_ in enumerate(click_track.notes):
                        if click_.start > note_.start:
                            break_point = step - 1
                            break
                    for i in range(break_point):
                        click_track.notes.pop(0)
            elif isinstance(msg, str):
                if msg == 'start_playing':
                    start_playing = True

        if start_playing:
            piano_rel = deepcopy(piano)
            click_rel = deepcopy(click_track)

            note_ = piano_rel.notes[0]
            break_point = 0
            for step, click_ in enumerate(click_rel.notes):
                if click_.start > note_.start:
                    break_point = step - 1
                    break
            for i in range(break_point):
                click_rel.notes.pop(0)

            org_st_time = click_rel.notes[0].start
            timing_queue.put(org_st_time)
            for note in piano_rel.notes:
                note.start -= org_st_time
            for note in click_rel.notes:
                note.start -= org_st_time

            pm = pretty_midi.PrettyMIDI()
            time_sig = pretty_midi.containers.TimeSignature(numerator=3, denominator=4, time=0)
            pm.time_signature_changes.append(time_sig)

            pm.instruments.append(piano_rel)
            pm.instruments.append(click_rel)

            x = get_features(pm)
            batch = {k:((v[None, :] if len(v.size()) == 1 else v) if isinstance(v,torch.Tensor) else v) for k,v in x.items()}

            batch_size, seq_len = batch['input_ids'].shape[:2]
            batch_ = { key: batch[key][:, :initial_prompt] for key in ['input_ids', 'bar_ids', 'position_ids'] }
            if model.description_flavor in ['description', 'both']:
                batch_['description'] = batch['description']
                batch_['desc_bar_ids'] = batch['desc_bar_ids']

            max_len = seq_len + 1024
            if max_iter > 0:
                max_len = min(max_len, initial_prompt + max_iter)

            pad_token_id = model.vocab.to_i(PAD_TOKEN)
            eos_token_id = model.vocab.to_i(EOS_TOKEN)

            batch_size, curr_len = batch_['input_ids'].shape

            i = curr_len - 1
            x = batch_['input_ids']
            player_buffer = []
            timer_ = 0.0
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
                if MAX_BARS > 0:
                    is_done.masked_fill_(next_bar_ids >= MAX_BARS + 1, True)

                x = torch.cat([x, next_token_ids.clone().unsqueeze(1)], dim=1)
                player_buffer.extend(model.vocab.decode(next_token_ids.clone().detach().cpu()))
                AI_out_queue.put(player_buffer)

                bar_ids = torch.cat([bar_ids, next_bar_ids.unsqueeze(1)], dim=1)
                position_ids = torch.cat([position_ids, next_position_ids.unsqueeze(1)], dim=1)

                if torch.all(is_done):
                    start_playing = False
                    break

            played_notes = []
            played_clicks = []
            decoded = model.vocab.decode(x.clone().detach().cpu()[0])
            pm = pretty_midi.PrettyMIDI()
            time_sig = pretty_midi.containers.TimeSignature(numerator=3, denominator=4, time=0)
            pm.time_signature_changes.append(time_sig)
            piano_x = pretty_midi.Instrument(program=0)
            click_track_x = pretty_midi.Instrument(program=9)

            for i in range(len(decoded)):
                pm = remi2midi(decoded[:i])
                for instrument in pm.instruments:
                    if instrument.program == 0:
                        for note in instrument.notes:
                            if not any(all([note.start == inst.start, note.end == inst.end, note.pitch == inst.pitch, note.velocity == inst.velocity]) for inst in played_notes):
                                played_notes.append(note)
                                piano_x.notes.append(note)
                    if instrument.program == 9:
                        for note in instrument.notes:
                            if not any(all([note.start == inst.start, note.end == inst.end, note.pitch == inst.pitch, note.velocity == inst.velocity]) for inst in played_clicks):
                                played_clicks.append(note)
                                click_track_x.notes.append(note)

            pm.instruments.append(piano_x)
            pm.instruments.append(click_track_x)

            pm.write('./res.midi')
        time.sleep(0.01)

def AI_client():
    while True:
        if not timing_queue.empty():
            org_timing_offset = timing_queue.get()
            print(org_timing_offset)
            played_notes = []
        if not AI_out_queue.empty():
            player_buffer = AI_out_queue.get()
            converted_ = remi2midi(player_buffer)
            # This needs to be fixed properly
            for instrument in converted_.instruments:
                if instrument.program == 0:
                    for note in instrument.notes:
                        if not any(all([
                            note.start == inst.start, 
                            note.end == inst.end, 
                            note.pitch == inst.pitch, 
                            note.velocity == inst.velocity
                        ]) for inst in played_notes):
                            transport_in_queue.put((note.start + org_timing_offset, 'note_on', note))
                            transport_in_queue.put((note.end + org_timing_offset, 'note_off', note))
                            played_notes.append(note)
        time.sleep(0.01)

if __name__ == "__main__":
    transport_server = td.Thread(target=transport)
    user_server = td.Thread(target=user_input)
    AI_server = td.Thread(target=AI_host)
    AI_client_server = td.Thread(target=AI_client)

    transport_server.start()
    user_server.start()
    AI_server.start()
    AI_client_server.start()

    transport_server.join()
    user_server.join()
    AI_server.join()
    AI_client_server.join()