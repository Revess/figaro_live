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
import queue
# TODO: add a cfg for all these settings like PPQN, ignored_clocks ctx size etc
# TODO: Fix note by note generation. It seems to almost work, otherwise just do bar by bar (but this is slower sadly...)
# TODO: Fix transport from the AI to the midi thread, seems still very buggy

notes_queue = queue.Queue()
click_queue = queue.Queue()
signals_queue = queue.Queue()
ai_queue = queue.Queue()

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'mps' if sys.platform == 'darwin' else 'cpu'
PPQN = 24
IGNORED_CLOCK_TICKS = 256
CTX_SIZE = 256
MAX_NOTES = 10
N_MEASURES = 2
bpm = 120
temperature = 0.8
initial_prompt = 1
max_bars = N_MEASURES
max_iter = 16000

print('getting outputs')
if __name__ == "__main__":
    MIDI_OUT = 'FLUID Synth (7786):Synth input port (7786:0) 130:0'
    MIDI_IN = 'Launchkey Mini MK3:Launchkey Mini MK3 Launchkey Mi 20:0'
    '''
    outputs, inputs = mido.get_output_names(), mido.get_input_names()
    while True:
        for i, o in enumerate(outputs):
            print(f'{i}:', o)
        opt = input(f'From the list of outputs above, which MIDI output device will be used?\t| {", ".join([str(x) for x in range(len(outputs))])} | >> ')
        if opt.isdigit():
            if int(opt) in range(len(outputs)):
                MIDI_OUT = outputs[int(opt)]
                print(f"Selected MIDI output device: {MIDI_OUT}\n")
                break
            else:
                print(f"Error, option {opt} is an invalid number...\n")
        else:
            print("Error, invalid choice please provide a number...\n")
    while True:
        for i, o in enumerate(inputs):
            print(f'{i}:', o)
        opt = input(f'From the list of outputs above, which MIDI input device will be used?\t| {", ".join([str(x) for x in range(len(inputs))])} | >> ')
        if opt.isdigit():
            if int(opt) in range(len(inputs)):
                MIDI_IN = inputs[int(opt)]
                print(f"Selected MIDI input device: {MIDI_IN}\n")
                break
            else:
                print(f"Error, option {opt} is an invalid number...\n")
        else:
            print("Error, invalid choice please provide a number...\n")
    '''

def external_midi_td():
    start_of_play = time.time()
    start_of_ai = 0
    next_ai_note = None

    tick = 0
    avg_delta = 0
    measure = 4
    beat = 0
    ignored_ticks = 0
    measure_count = 0
    start_playing = False

    piano = pretty_midi.Instrument(program=0)
    click_track = pretty_midi.Instrument(program=9)

    active_notes = {}

    notes_to_play = []

    with mido.open_input(MIDI_IN) as inport, mido.open_output(MIDI_OUT) as outport:
        print('using', MIDI_IN, MIDI_OUT)
        tick_start = time.time()
        for msg in inport:
            if not ai_queue.empty():
                notes_to_play.extend(ai_queue.get())

            if msg.type == 'clock':
                if ignored_ticks < IGNORED_CLOCK_TICKS:
                    ignored_ticks += 1
                    continue  # Skip processing until we reach stable timing
                delta = time.time() - tick_start
                avg_delta += delta
                if tick % PPQN == 0 and tick != 0:
                    tick = 0
                    bpm = round(60 / avg_delta, 0)
                    avg_delta = 0

                    time_ = time.time() - start_of_play
                    if beat % measure == 0 and beat != 0:
                        outport.send(mido.Message('note_on', note=75, velocity=100, channel=9))
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=75,
                            start=time_,
                            end=time_ + (60 / bpm)
                        )
                        click_track.notes.append(note)
                        click_queue.put(note)
                        time.sleep(0.01)
                        outport.send(mido.Message('note_off', note=75, velocity=100, channel=9))
                        beat = 0
                        if measure_count % N_MEASURES == 0 and measure_count != 0:
                            measure_count = 0
                            start_playing = not start_playing
                            signals_queue.put({'start_playing': start_playing})
                            start_of_ai = time.time()

                        measure_count += 1
                    else:
                        outport.send(mido.Message('note_on', note=56, velocity=100, channel=9))
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=56,
                            start=time_,
                            end=time_ + (60 / bpm)
                        )
                        click_track.notes.append(note)
                        click_queue.put(note)
                        time.sleep(0.01)
                        outport.send(mido.Message('note_off', note=56, velocity=100, channel=9))

                    beat += 1

                tick += 1
                tick_start = time.time()

            elif msg.type == 'note_on' and msg.velocity > 0 and not start_playing:
                outport.send(msg)
                active_notes[msg.note] = (time.time() - start_of_play, msg.velocity)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0) and not start_playing:
                outport.send(msg)
                if msg.note in active_notes:
                    start, velocity = active_notes.pop(msg.note)
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=msg.note,
                        start=start,
                        end=time.time() - start_of_play
                    )
                    piano.notes.append(note)
                    notes_queue.put(note) # Send the note to the model's thread
                    if len(piano.notes) > 2048:
                        piano.notes.pop(0)
            
            if start_playing:
                if len(notes_to_play) > 0 and next_ai_note is None:
                    next_ai_note = notes_to_play.pop(0)

                if next_ai_note is not None:
                    if next_ai_note[0] < (time.time() - start_of_ai):
                        msg = mido.Message(next_ai_note[1], note=note.pitch, velocity=note.velocity)
                        outport.send(msg)
                        next_ai_note = None

def model_td():
    start_play_toggle = False
    piano = pretty_midi.Instrument(program=0)
    click_track = pretty_midi.Instrument(program=9)

    model, vae_module = load_model('./checkpoints/figaro-expert.ckpt', './checkpoints/vq-vae.ckpt')
    model.to('cuda:0')

    while True:
        signal_k = ''
        signal_v = None

        if not notes_queue.empty():
            note_ = notes_queue.get(False)
            piano.notes.append(note_)
            if len(piano.notes) > MAX_NOTES:
                note_ = piano.notes.pop(0)
                break_point = 0
                for step, click_ in enumerate(click_track.notes):
                    if click_.start > note_.start:
                        break_point = step - 1
                        break
                for i in range(break_point):
                    click_track.notes.pop(0)
        if not click_queue.empty():
            note_ = click_queue.get(False)
            click_track.notes.append(note_)
        if not signals_queue.empty():
            signal = signals_queue.get(False)
            signal_k, signal_v = list(signal.items())[0]


        if signal_k == 'start_playing':
            start_play_toggle = signal_v

        if start_play_toggle:
            pm = pretty_midi.PrettyMIDI()
            time_sig = pretty_midi.containers.TimeSignature(numerator=3, denominator=4, time=0)
            pm.time_signature_changes.append(time_sig)

            pm.instruments.append(piano)
            pm.instruments.append(click_track)

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
                if max_bars > 0:
                    is_done.masked_fill_(next_bar_ids >= max_bars + 1, True)

                x = torch.cat([x, next_token_ids.clone().unsqueeze(1)], dim=1)
                player_buffer.extend(model.vocab.decode(next_token_ids.clone().detach().cpu()))
                # player_buffer = torch.cat([player_buffer, next_token_ids.clone()], dim=0)
                converted_ = remi2midi(player_buffer)
                played_note = False
                events = []
                if len(converted_.instruments) > 0:
                    for instrument in converted_.instruments:
                        if len(instrument.notes) > 0:
                            for note in instrument.notes:
                                if note.start >= timer_:
                                    played_note = True
                                    note_end = timer_ + (note.end - note.start)
                                    note_start = timer_ + note.start
                                    timer_ = note_start
                                    events.append((note_start, 'note_on', note))
                                    events.append((note_end, 'note_off', note))
                events.sort(key=lambda event: event[0])
                print(events)

                if len(events) > 0:
                    ai_queue.put(events)

                bar_ids = torch.cat([bar_ids, next_bar_ids.unsqueeze(1)], dim=1)
                position_ids = torch.cat([position_ids, next_position_ids.unsqueeze(1)], dim=1)

                if torch.all(is_done):
                    break
                # print()

        time.sleep(0.01)

if __name__ == "__main__":
    midi_server = td.Thread(target=external_midi_td)
    model_server = td.Thread(target=model_td)

    midi_server.start()
    model_server.start()

    midi_server.join()
    model_server.join()