import mido
import pretty_midi
import multiprocessing
import time
import json
import torch
from figaro.live_utils import load_model, get_features, sample
from figaro.input_representation import remi2midi
from copy import deepcopy

# TODO: add output of the model to the input loop.
# Check out notochord, maybe it is better and more useful.
# 

def transport(send_pipe, recv_pipe):
    with open('./cfg.json', 'r') as f:
        cfg = json.load(f)
    MIDI_IN = cfg['MIDI_IN']
    MIDI_OUT = cfg['MIDI_OUT']
    PPQN = cfg['PPQN']
    METER = cfg['METER']
    MAX_BARS = cfg['MAX_BARS']
    VERBOSE = cfg['VERBOSE']
    CLOCK_IN = cfg['CLOCK_IN']

    CLOCK_OUT_CHAN = cfg['CLOCK_OUT_CHAN']
    AI_OUT_CHAN = cfg['AI_OUT_CHAN']
    MUSICIAN_OUT_CHAN = cfg['MUSICIAN_OUT_CHAN']

    start_time = time.time()
    tick_start = time.time()
    tick = avg_delta = beat = 0
    active_notes = {}
    active_ai_notes = {}
    bars_played_user = 0
    bars_played_ai = 0
    send_to_ai = False
    notes_to_play = []
    sort = False
    ai_turn = False

    with mido.open_input(MIDI_IN) as inport, mido.open_output(MIDI_OUT) as outport:
        for msg in inport:
            if recv_pipe.poll(0.001):
                res = recv_pipe.recv()
                if isinstance(res, str):
                    if res == 'done':
                        send_to_ai = False
                        bars_played_user = 0
                elif isinstance(res, tuple):
                    if ai_turn:
                        notes_to_play.append(res)
                    sort = True

            if sort:
                notes_to_play.sort(key=lambda event: event[0])
                sort = False

            if msg.type == 'clock':
                delta = time.time() - tick_start
                avg_delta += delta

                if (tick + 1) % PPQN == 0:
                    tick = 0
                    bpm = round(60 / avg_delta, 0)
                    avg_delta = 0

                    time_ = time.time() - start_time
                    if beat % METER == 0:
                        pitch = 75
                        if bars_played_user >= 1:
                            bars_played_user += 1
                        if bars_played_ai >= 1:
                            bars_played_ai += 1
                    else:
                        pitch = 56

                    if beat % METER == 0 and beat != 0:
                        beat = 0
                    beat += 1

                    # Sending notes
                    if VERBOSE:
                        print('click', bars_played_ai, bars_played_user)
                        outport.send(mido.Message('note_on', note=pitch, velocity=100, channel=CLOCK_OUT_CHAN))
                        time.sleep(0.01)
                        outport.send(mido.Message('note_off', note=pitch, velocity=100, channel=CLOCK_OUT_CHAN))

                    
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=time_,
                        end=time_ + (60 / bpm)
                    )

                    send_pipe.send(('click', note))

                tick += 1
                tick_start = time.time()

            if bars_played_user == MAX_BARS + 1 and not send_to_ai:
                send_pipe.send('start_ai')
                send_to_ai = True
                ai_turn = True
            
            if bars_played_ai == MAX_BARS + 1:
                bars_played_ai = 0 
                for note_ in notes_to_play[:]:
                    if note_[1] == 'note_off' and note_[-1].pitch in active_ai_notes.keys():
                        start, velocity = active_ai_notes.pop(note_[-1].pitch)
                        outport.send(mido.Message(note_[1], note=note_[-1].pitch, velocity=note_[-1].velocity, channel=AI_OUT_CHAN))
                        notes_to_play.remove(note_)
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=note_[-1].pitch,
                            start=start,
                            end=(time.time() - start_time) 
                        )

                        send_pipe.send(('piano', note))
                notes_to_play = []
                ai_turn = False
                
            if msg.type == 'note_on' and msg.velocity > 0 and not send_to_ai:
                bars_played_ai = 0
                outport.send(msg)
                active_notes[msg.note] = ((time.time() - start_time), msg.velocity)
                if bars_played_user == 0:
                    bars_played_user += 1

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0) and msg.note in active_notes.keys() and not send_to_ai:
                outport.send(msg)
                start, velocity = active_notes.pop(msg.note)
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=msg.note,
                    start=start,
                    end=(time.time() - start_time) 
                )

                send_pipe.send(('piano', note))

            if ai_turn:
                for note_ in notes_to_play[:]:
                    if note_[0] < (time.time() - start_time):
                        if note_[1] == 'note_on' and note_[-1].velocity > 0:
                            active_ai_notes[note_[-1].pitch] = ((time.time() - start_time), note_[-1].velocity)
                            outport.send(mido.Message(note_[1], note=note_[-1].pitch, velocity=note_[-1].velocity, channel=AI_OUT_CHAN))
                            notes_to_play.remove(note_)
                        if note_[1] == 'note_off' and note_[-1].pitch in active_ai_notes.keys():
                            start, velocity = active_ai_notes.pop(note_[-1].pitch)
                            outport.send(mido.Message(note_[1], note=note_[-1].pitch, velocity=note_[-1].velocity, channel=AI_OUT_CHAN))
                            notes_to_play.remove(note_)
                            note = pretty_midi.Note(
                                velocity=velocity,
                                pitch=note_[-1].pitch,
                                start=start,
                                end=(time.time() - start_time) 
                            )

                            send_pipe.send(('piano', note))
                        if bars_played_ai == 0:
                            bars_played_ai += 1

def AI_note_manager(send_transport, recv_transport, send_AI, receive_AI):
    piano_notes = []
    click_notes = []
    ORG_OFFSET = 0.0
    get_next_offset = False
    played_notes = []
    while True:
        if receive_AI.poll(0.001):
            data = receive_AI.recv()
            if isinstance(data, str):
                if data == 'done':
                    send_transport.send('done')
            elif isinstance(data, list):
                converted_ = remi2midi(data)
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
                                send_transport.send((note.start + ORG_OFFSET, 'note_on', note))
                                send_transport.send((note.end + ORG_OFFSET, 'note_off', note))
                                played_notes.append(note)
        if recv_transport.poll(0.001):
            data = recv_transport.recv()
            if isinstance(data, str):
                if data == 'start_ai':
                    get_next_offset = True
                    # Copy the notes
                    played_notes = []

                    piano_rel = deepcopy(piano_notes)
                    click_rel = deepcopy(click_notes)
                
                    # Do one last time filter round
                    break_point = 0
                    for step, click_ in enumerate(click_rel):
                        if click_.start > piano_rel[0].start:
                            break_point = step - 1
                            break
                    for i in range(break_point):
                        click_rel.pop(0)

                    piano = pretty_midi.Instrument(program=0)
                    click_track = pretty_midi.Instrument(program=9)

                    for note in piano_rel:
                        note.start -= ORG_OFFSET
                        piano.notes.append(note)
                    for note in click_rel:
                        note.start -= ORG_OFFSET
                        click_track.notes.append(note)

                    pm = pretty_midi.PrettyMIDI()
                    time_sig = pretty_midi.containers.TimeSignature(numerator=4, denominator=4, time=0)
                    pm.time_signature_changes.append(time_sig)

                    pm.instruments.append(piano)
                    pm.instruments.append(click_track)

                    send_AI.send(pm)
                # send_transport.send('done')
            elif isinstance(data, tuple):
                if data[0] == 'click':
                    if get_next_offset:
                        ORG_OFFSET = data[1].start
                        get_next_offset = False
                    click_notes.append(data[1])
                else:
                    piano_notes.append(data[1])
                    piano_notes = piano_notes[-256:] # Filter so the piano_notes at most get a ctx of -256
                    break_point = 0
                    for step, click_ in enumerate(click_notes):
                        if click_.start > piano_notes[0].start:
                            break_point = step - 1
                            break
                    for i in range(break_point):
                        click_notes.pop(0)
            # print(len(piano_notes), len(click_notes), click_notes[-1].start)

def AI_Processor(send_pipe, recv_pipe):
    from figaro.constants import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, BAR_KEY, POSITION_KEY

    with open('./cfg.json', 'r') as f:
        cfg = json.load(f)
    MAX_BARS = cfg['MAX_BARS']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
    temperature = cfg['TEMPERATURE']
    initial_prompt = 1
    max_iter = 16000

    model, vae_module = load_model('./checkpoints/figaro-expert.ckpt', './checkpoints/vq-vae.ckpt')
    model.to(DEVICE)

    while True:
        pm = recv_pipe.recv()
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

            # Send to processor
            send_pipe.send(player_buffer)

            bar_ids = torch.cat([bar_ids, next_bar_ids.unsqueeze(1)], dim=1)
            position_ids = torch.cat([position_ids, next_position_ids.unsqueeze(1)], dim=1)

            if torch.all(is_done):
                send_pipe.send('done')
                break

        played_notes = []
        played_clicks = []
        decoded = model.vocab.decode(x.clone().detach().cpu()[0])
        
        pm = pretty_midi.PrettyMIDI()
        time_sig = pretty_midi.containers.TimeSignature(numerator=4, denominator=4, time=0)
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

if __name__ == "__main__":
    p1_to_p2_recv, p1_to_p2_send = multiprocessing.Pipe(duplex=False)
    p2_to_p1_recv, p2_to_p1_send = multiprocessing.Pipe(duplex=False)
    p2_to_p3_recv, p2_to_p3_send = multiprocessing.Pipe(duplex=False)
    p3_to_p2_recv, p3_to_p2_send = multiprocessing.Pipe(duplex=False)
    
    # Set up processes with appropriate pipe ends:
    p1 = multiprocessing.Process(target=transport, args=(p1_to_p2_send, p2_to_p1_recv))
    p2 = multiprocessing.Process(target=AI_note_manager, args=(p2_to_p1_send, p1_to_p2_recv, p2_to_p3_send, p3_to_p2_recv))
    p3 = multiprocessing.Process(target=AI_Processor, args=(p3_to_p2_send, p2_to_p3_recv))
    
    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    p2.join()
    p3.join()