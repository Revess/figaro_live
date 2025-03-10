import mido
import pretty_midi
import multiprocessing
import time
import json
import torch
import re
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

@dataclass
class Note():
    start: float
    end: float
    pitch: int
    velocity: int
    channel: int

def conv_channel(channel):
    channel_map = {
        0: "%",
        1: "^",
        2: "&",
        3: "*",
        4: ";",
        5: ":",
        6: "'",
        7: '"',
        9: ")",
        10: "{",
        11: "}",
        12: "[",
        13: "]",
        14: "(",
    }

    if isinstance(channel, int):
        if channel in channel_map.keys():
            return channel_map[channel]
        return "%"
    elif isinstance(channel, str):
        # Invert the mapping
        channel_map = {
            v: k for k, v in channel_map.items()
        }
        if channel in channel_map.keys():
            return channel_map[channel]
        return 0
    else:
        raise "Wrong type for channel"

def conv_velocity(velocity):
    velocity_map = {
        48: "!",
        60: "@",
        100: "#",
    }
    if isinstance(velocity, int):
        for i in velocity_map.keys():
            if velocity <= i:
                return velocity_map[i]
        return "@"
    elif isinstance(velocity, str):
        # Invert the mapping
        velocity_map = {
            v: k for k, v in velocity_map.items()
        }
        return velocity_map[velocity]
    else:
        raise "Wrong type for velocity"

def conv_pm_to_str(midi_):
    ticks_per_beat = 480 / 24
    bpm = ((60) / 120) *  1000000

    rel_notes = []
    last_time = 0

    for msg in midi_:
        delta = msg.end - msg.start
        rel_notes.append(
            str(mido.second2tick(msg.start - last_time, ticks_per_beat, bpm)) +
            str(conv_velocity(msg.velocity)) +
            str(mido.second2tick(delta, ticks_per_beat, bpm)) +
            str(conv_channel(msg.channel)) +
            str(msg.pitch) + "|"
        )

        last_time = msg.start

    str_conv = ''.join(rel_notes)
    return str_conv

def transport(send_pipe, recv_pipe):
    with open('./cfg.json', 'r') as f:
        cfg = json.load(f)
    MIDI_IN = cfg['MIDI_IN']
    MIDI_OUT = cfg['MIDI_OUT']
    PPQN = cfg['PPQN']
    METER = cfg['METER']
    MAX_BARS = cfg['MAX_BARS']
    VERBOSE = cfg['VERBOSE']
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
                elif isinstance(res, Note):
                    if ai_turn:
                        notes_to_play.append((
                            res.start,
                            'note_on',
                            note
                        ))
                        notes_to_play.append((
                            res.end,
                            'note_of',
                            note
                        ))
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
                        outport.send(mido.Message('note_on', note=pitch, velocity=100, channel=9))
                        time.sleep(0.01)
                        outport.send(mido.Message('note_off', note=pitch, velocity=100, channel=9))

                    
                    note = Note(
                        velocity=100,
                        pitch=pitch,
                        start=mido.second2tick(time_, 480 / PPQN, bpm),
                        end=mido.second2tick(time_ + (60 / bpm), 480 / PPQN, bpm),
                        channel=9
                    )

                    send_pipe.send(('click', note))

                tick += 1
                tick_start = time.time()

            if bars_played_user == MAX_BARS + 1 and not send_to_ai:
                send_pipe.send('start_ai')
                send_to_ai = True
                ai_turn = True
            
            if bars_played_ai == MAX_BARS + 1:
                send_pipe.send('bar_end')
                bars_played_ai = 0 
                for note_ in notes_to_play[:]:
                    if note_[1] == 'note_off' and note_[-1].velocity == 0 and note_[-1] in active_ai_notes.keys():
                        start, velocity = active_notes.pop(note_[-1].pitch)
                        outport.send(mido.Message(note_[1], note=note_[-1].pitch, velocity=note_[-1].velocity))
                        notes_to_play.remove(note_)
                        note = Note(
                            velocity=velocity,
                            pitch=msg.note,
                            start=mido.second2tick(start, 480 / PPQN, bpm),
                            end=mido.second2tick((time.time() - start_time), 480 / PPQN, bpm),
                            channel=0
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
                note = Note(
                    velocity=velocity,
                    pitch=msg.note,
                    start=mido.second2tick(start, 480 / PPQN, bpm),
                    end=mido.second2tick((time.time() - start_time), 480 / PPQN, bpm),
                    channel=0
                )

                send_pipe.send(('piano', note))

            if ai_turn:
                for note_ in notes_to_play[:]:
                    if note_[0] < (time.time() - start_time):
                        if note_[1] == 'note_on' and note_[-1].velocity > 0:
                            active_ai_notes[note_[-1].pitch] = ((time.time() - start_time), note_[-1].velocity)
                            outport.send(mido.Message(note_[1], note=note_[-1].pitch, velocity=note_[-1].velocity))
                            notes_to_play.remove(note_)
                        if note_[1] == 'note_off' and note_[-1].velocity == 0 and note_[-1] in active_ai_notes.keys():
                            start, velocity = active_notes.pop(note_[-1].pitch)
                            outport.send(mido.Message(note_[1], note=note_[-1].pitch, velocity=note_[-1].velocity))
                            notes_to_play.remove(note_)
                            note = Note(
                                velocity=velocity,
                                pitch=msg.note,
                                start=mido.second2tick(start, 480 / PPQN, bpm),
                                end=mido.second2tick((time.time() - start_time), 480 / PPQN, bpm),
                                channel=0
                            )

                            send_pipe.send(('piano', note))
                        if bars_played_ai == 0:
                            bars_played_ai += 1

def AI_note_manager(send_transport, recv_transport, send_AI, receive_AI):
    piano_notes = []
    click_notes = []
    ORG_OFFSET = 0.0
    init_offset = 0
    get_next_offset = False
    while True:
        if receive_AI.poll(0.001):
            data = receive_AI.recv()
            if isinstance(data, Note):
                if data.channel == 0:
                    data.start += init_offset
                    data.end += data.start
                    init_offset = data.start
                    send_transport.send(data)
        if recv_transport.poll(0.001):
            data = recv_transport.recv()
            if isinstance(data, str):
                if data == 'start_ai':
                    init_offset = ORG_OFFSET
                    get_next_offset = True
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

                    piano_track = []
                    click_track = []

                    for note in piano_rel:
                        note.start -= ORG_OFFSET
                        piano_track.append(note)
                    for note in click_rel:
                        note.start -= ORG_OFFSET
                        click_track.append(note)

                    midi_ = []
                    for note in piano_track:
                        midi_.append(Note(
                            start = note.start,
                            end = note.end,
                            pitch = note.pitch,
                            velocity = note.velocity,
                            channel = 0
                        ))
                    for note in click_track:
                        midi_.append(Note(
                            start = note.start,
                            end = note.end,
                            pitch = note.pitch,
                            velocity = note.velocity,
                            channel = 9
                        ))

                    midi_.sort(key = lambda x: x.start)
                    pm = conv_pm_to_str(midi_)
                    send_AI.send(pm)
                elif data == 'bar_end':
                    send_AI('done')
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
    with open('./cfg.json', 'r') as f:
        cfg = json.load(f)
    MAX_BARS = cfg['MAX_BARS']
    model_name = "kobimusic/esecutore-4-0619"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
    temperature = cfg['TEMPERATURE']
    initial_prompt = 1
    max_iter = 16000
    max_notes_ctx = 512

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('mps')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(DEVICE)

    while True:
        tokens = recv_pipe.recv()
        print(tokens)
        while True:
            if recv_pipe.poll(0.001):
                data = recv_pipe.recv()
                if data == 'done':
                    break
            tokens = f". pop |{tokens}"
            ins = tokenizer.encode(tokens)
            ins = torch.tensor([ins], device=DEVICE)
            res = model.generate(
                ins,
                use_cache=False,
                max_new_tokens=6,
                do_sample=True,
                temperature=0.89,
                top_p=1.0,
                num_return_sequences=1,
            )
            decoded = tokenizer.batch_decode(res[:, ins.shape[1]:])[0]
            print(decoded)
            pattern = re.compile(r"(\d+)(\D)(\d+)(\D)(\d+)\|")
            m = pattern.match(decoded)
            decoded = Note(
                start = int(m.group(1)),
                velocity = conv_velocity(m.group(2)),
                end = int(m.group(3)),
                channel = conv_channel(m.group(4)),
                pitch = int(m.group(5))
            )
            send_pipe.send(decoded)
            ins = torch.cat((ins, res[:, ins.shape[1]:]), dim=1)
            if ins.shape[1] > max_notes_ctx:
                ins = ins[:, -max_notes_ctx:]

            

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