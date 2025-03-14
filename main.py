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

import transformers
import warnings

warnings.filterwarnings("ignore", message=".*attention mask and the pad token id.*")
transformers.logging.set_verbosity_error()

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
        127: "$",
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

def transport(send_pipe, recv_pipe):
    with open('./cfg.json', 'r') as f:
        cfg = json.load(f)
    MIDI_IN = cfg['MIDI_IN']
    MIDI_OUT = cfg['MIDI_OUT']
    PPQN = cfg['PPQN']
    METER = cfg['METER']
    MAX_BARS = cfg['MAX_BARS']
    VERBOSE = cfg['VERBOSE']

    abs_time = None
    tick_start = None

    active_notes = {}
    active_ai_notes = {}

    active_player_bars = 0
    active_ai_bars = -1
    last_tick_time = 0
    notes_to_play = []
    tempo_mics = 1

    with mido.open_input(MIDI_IN) as inport, mido.open_output(MIDI_OUT) as outport:
        avg_delta = 0
        tick = 0
        beat = 0
        for msg in inport:
            if recv_pipe.poll(0.001):
                data = recv_pipe.recv()
                notes_to_play.extend(data)
                notes_to_play.sort(key = lambda x: x.time)
            # Handle pipe inputs
            if msg.type == 'clock':
                if abs_time is None:
                    abs_time = time.time()
                if tick_start is None:
                    tick_start = time.time()

                delta = time.time() - tick_start
                avg_delta += delta

                if (tick + 1) % PPQN == 0:
                    tick = 0
                    bpm = round(60 / avg_delta, 0)
                    tempo_mics = (60 / bpm) * 1000000
                    avg_delta = 0

                    time_ = time.time() - abs_time
                    if beat % METER == 0:
                        pitch = 75
                        last_tick_time = mido.second2tick(time_, PPQN, (60 / bpm) * 1000000) - PPQN
                        if active_player_bars > 0:
                            active_player_bars += 1
                        if active_ai_bars >= 0:
                            active_ai_bars += 1
                    else:
                        pitch = 56

                    if beat % METER == 0 and beat != 0:
                        beat = 0
                    beat += 1

                    # Sending notes
                    if VERBOSE:
                        outport.send(mido.Message('note_on', note=pitch, velocity=100, channel=9))
                        time.sleep(0.01)
                        outport.send(mido.Message('note_off', note=pitch, velocity=100, channel=9))
                    
                    note = Note(
                        velocity=100,
                        pitch=pitch,
                        start=mido.second2tick(
                            time_, 
                            PPQN, 
                            tempo_mics
                        ),
                        end=mido.second2tick(
                            time_ + (60 / bpm), 
                            PPQN, 
                            tempo_mics
                        ),
                        channel=9
                    )

                    send_pipe.send(('click', note))

                tick += 1
                tick_start = time.time()

            if active_player_bars == MAX_BARS + 1:
                for note, (start, velocity) in active_notes.items():
                    note = Note(
                        velocity=velocity,
                        pitch=note,
                        start=mido.second2tick(
                            start, 
                            PPQN, 
                            tempo_mics
                        ),
                        end=mido.second2tick(
                            time.time() - abs_time, 
                            PPQN, 
                            tempo_mics
                        ),
                        channel=0
                    )

                    send_pipe.send(('piano', note))
                send_pipe.send(('start_ai', last_tick_time))
                active_ai_bars = 0
                notes_to_play = []
                active_player_bars = -1
            
            if active_ai_bars == MAX_BARS + 1:
                send_pipe.send(('stop_ai', ''))
                for note_ in notes_to_play[:]:
                    if note_.type == 'note_off' and note_.velocity == 0 and note_.note in active_ai_notes.keys():
                        start, velocity = active_notes.pop(note_.note)
                        outport.send(mido.Message(note_.type, note=note_.note, velocity=note_.velocity))
                        notes_to_play.remove(note_)
                        note = Note(
                            velocity=velocity,
                            pitch=msg.note,
                            start=mido.second2tick(start, PPQN, tempo_mics),
                            end=mido.second2tick((time.time() - abs_time), PPQN, tempo_mics),
                            channel=0
                        )

                        send_pipe.send(('piano', note))
                active_ai_bars = -1
                active_player_bars = 0
                notes_to_play = []

            if msg.type in ['note_on', 'note_off']:

                if msg.type == 'note_on' and msg.velocity > 0 and active_player_bars >= 0 and active_player_bars <= MAX_BARS:
                    active_notes[msg.note] = (time.time() - abs_time, msg.velocity)
                    outport.send(msg)
                    if active_player_bars == 0:
                        active_player_bars += 1

                elif msg.type == 'note_off' or msg.velocity == 0 and msg.note in active_notes and active_player_bars >= 0 and active_player_bars <= MAX_BARS:
                    outport.send(msg)
                    start, velocity = active_notes.pop(msg.note)
                    note = Note(
                        velocity=velocity,
                        pitch=msg.note,
                        start=mido.second2tick(
                            start, 
                            PPQN, 
                            tempo_mics
                        ),
                        end=mido.second2tick(
                            time.time() - abs_time, 
                            PPQN, 
                            tempo_mics
                        ),
                        channel=0
                    )

                    send_pipe.send(('piano', note))

            if active_ai_bars >= 0:
                for note_ in notes_to_play[:]:
                    if note_.time < (time.time() - abs_time):
                        if note_.type == 'note_on' and note_.velocity > 0:
                            active_ai_notes[note_.note] = ((time.time() - abs_time), note_.velocity)
                            outport.send(mido.Message(note_.type, note=note_.note, velocity=note_.velocity, channel=1))
                            notes_to_play.remove(note_)
                        if note_.type == 'note_off' and note_.note in active_ai_notes.keys():
                            start, velocity = active_ai_notes.pop(note_.note)
                            outport.send(mido.Message(note_.type, note=note_.note, velocity=note_.velocity, channel=1))
                            notes_to_play.remove(note_)
                            note = Note(
                                velocity=velocity,
                                pitch=note_.note,
                                start=mido.second2tick(start, PPQN, tempo_mics),
                                end=mido.second2tick((time.time() - abs_time), PPQN, tempo_mics),
                                channel=0
                            )

                            send_pipe.send(('piano', note))

def AI_note_manager(send_transport, recv_transport, send_AI, receive_AI):
    piano_notes = []
    click_notes = []
    max_notes_ctx = 128
    last_tick_time = 0

    while True:
        if receive_AI.poll(0.001):
            decoded = receive_AI.recv()

            print(decoded)

            mido_stack = []
            for token in decoded.split('|'):
                pattern = re.compile(r"(\d+)(\D)(\d+)(\D)(\d+)")
                m = pattern.match(token)
                if m:
                    if conv_channel(m.group(4)) == 0:
                        st_time = mido.tick2second(int(m.group(1)) + last_tick_time, 480, 500000)
                        mido_stack.append(mido.Message(
                            'note_on', note = int(m.group(5)), velocity = conv_velocity(m.group(2)), time = st_time
                        ))

                        mido_stack.append(mido.Message(
                            'note_off', note = int(m.group(5)), velocity = conv_velocity(m.group(2)), time = st_time + mido.tick2second(int(m.group(3)), 480, 500000)
                        ))
                    
                        last_tick_time += st_time
            print(mido_stack)
            if len(mido_stack) > 0:
                send_transport.send(mido_stack)

        if recv_transport.poll(0.001):
            k, v = recv_transport.recv()
            if k == 'click':
                click_notes.append(v)
            elif k == 'piano':
                br = False
                for note in piano_notes:
                    if note.pitch == v.pitch and note.velocity == v.velocity and note.start == v.start and note.channel == v.channel and note.end < v.end: # Case where the actual note end is different
                        print('found one!')
                        note.end = v.end
                        br = True
                        break
                if not br:
                    piano_notes.append(v)
            elif k == 'stop_ai':
                send_AI.send('stop_ai')
                piano_notes = []
                click_notes = []
            elif k == 'start_ai':
                last_tick_time = v
                piano_notes.sort(key = lambda x: x.start)
                click_notes.sort(key = lambda x: x.start)
                piano_notes = piano_notes[-max_notes_ctx:]

                break_point = 0
                for step, click_ in enumerate(click_notes):
                    if click_.start > piano_notes[0].start:
                        break_point = step - 1
                        break
                for i in range(break_point):
                    click_notes.pop(0)

                piano_rel = deepcopy(piano_notes)
                click_rel = deepcopy(click_notes)

                for note in piano_rel: # Move both abs points over !
                    note.start -= click_rel[0].start
                    note.end -= click_rel[0].start
                # for note in click_rel:
                #     note.start -= click_rel[0].start
                #     note.end -= click_rel[0].start

                midi_ = []
                for note in piano_rel:
                    midi_.append(Note(
                        start = note.start,
                        end = note.end,
                        pitch = note.pitch,
                        velocity = note.velocity,
                        channel = 0
                    ))
                # test without the clicks
                # for note in click_rel:
                #     midi_.append(Note(
                #         start = note.start,
                #         end = note.end,
                #         pitch = note.pitch,
                #         velocity = note.velocity,
                #         channel = 9
                #     ))

                midi_.sort(key = lambda x: x.start)

                midi_str = '. classical |'
                prev_start = 0
                for note in midi_:
                    midi_str += str(note.start - prev_start) + str(conv_velocity(note.velocity)) + str(note.end - note.start) + str(conv_channel(note.channel)) + str(note.pitch) + "|"
                    prev_start = note.start

                def convert_ppqn(data, factor=20):
                    # This regex matches a delimiter ($, @, #, or !) followed by one or more digits
                    def replace_tick(match):
                        symbol = match.group(1)
                        tick = int(match.group(2))
                        return f"{symbol}{int(tick * factor)}"
                    return re.sub(r'([$@#!])(\d+)', replace_tick, data)
                midi_str = convert_ppqn(midi_str, factor = 480 // 24)

                # with open('./test.txt', 'w') as f:
                #     f.write(midi_str)
                
                send_AI.send(midi_str)
        time.sleep(0.001)

def AI_Processor(send_pipe, recv_pipe):
    with open('./cfg.json', 'r') as f:
        cfg = json.load(f)
    model_name = "kobimusic/esecutore-4-0619"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
    temperature = cfg['TEMPERATURE']
    max_notes_ctx = 512

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('mps')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(DEVICE)
    print('loaded')

    while True:
        tokens = recv_pipe.recv()
        decoded_tokens = tokenizer.encode(tokens)
        ins = torch.tensor([decoded_tokens], device=DEVICE)
        attention_mask = torch.ones_like(ins)
        generated_notes = 0
        played_ = 0
        i = 0

        while True:
            if recv_pipe.poll(0.001):
                data = recv_pipe.recv()
                print(data)
                if data == 'stop_ai':
                    break
            res = model.generate(
                ins[:, i:],
                attention_mask = attention_mask,
                use_cache=False,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.89,
                top_p=1.0,
                num_return_sequences=1,
            )
            ins = torch.cat((ins, res[:, -1][:, None]), dim=1)
            i += 1

            coll = tokenizer.batch_decode(ins[:, len(decoded_tokens) + played_:].cpu().detach())[0]

            if '|' in coll:
                generated_notes += 1
                print(coll)
                send_pipe.send(coll)
                played_ += ins[:, len(decoded_tokens) + played_:].cpu().detach().shape[1]




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