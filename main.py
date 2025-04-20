import mido
import pretty_midi
import multiprocessing
import time
import json
import torch
import math
import os
from figaro.live_utils import load_model, get_features, sample
from figaro.input_representation import remi2midi
from copy import deepcopy

# TODO: add output of the model to the input loop.
# Check out notochord, maybe it is better and more useful.
# 

# TODO:
# - Find a clever way to sync Figaro to the clock
# - Send out triggers for listening

# with mido.open_input('Teensy MIDI Port 1') as clock_port, mido.open_input('Launchkey Mini MK3 MIDI Port') as inport, mido.open_output('IAC Driver Bus 1') as outport:
#     while True:
#         for port in (clock_port, inport):
#             for msg in port.iter_pending():
#                 print(msg, port.name)
#                 outport.send(msg)
#         time.sleep(0.001)  # small sleep to prevent high CPU usage

def calculate_certainty(logits, temperature=1.0):
    # Softmax over logits
    pr = torch.nn.functional.softmax(logits / temperature, dim=-1)
    
    # Compute entropy (uncertainty)
    entropy = -torch.sum(pr * torch.log(pr + 1e-10), dim=-1)  # add small epsilon to avoid log(0)
    
    # Certainty is the inverse of entropy
    certainty = 1.0 - (entropy / torch.log(torch.tensor(pr.size(-1), dtype=torch.float32)))
    
    return certainty


def map_value(input_value, input_min=60, input_max=170, output_min=0, output_max=127, prec = 'int'):
    scaled_value = ((input_value - input_min) * (output_max - output_min)) / (input_max - input_min) + output_min
    if prec == 'int':
        return int(scaled_value)
    else:
        return scaled_value

def reverse_map_value(scaled_value, input_min=60, input_max=170, output_min=0, output_max=127):
    input_value = ((scaled_value - output_min) * (input_max - input_min)) / (output_max - output_min) + input_min
    return input_value

def scale_note_times(note, scaling_factor):
    note.start *= scaling_factor
    note.end *= scaling_factor
    return note

def scale_back_note_times(note, scaling_factor):
    note.start /= scaling_factor
    note.end /= scaling_factor
    return note

def get_prompt(filepath, target_ppqn, target_bpm, meter = 4):
    """
    Reads a MIDI file, extracts notes, rescales timing to target PPQN and BPM,
    generates a click track, and calculates the start time of the next measure.

    Args:
        filepath (str): Path to the MIDI file.
        target_ppqn (int): The desired Pulses Per Quarter Note for output interpretation.
        target_bpm (float): The desired Beats Per Minute for output interpretation.
        meter (int): The number of beats per measure (e.g., 4 for 4/4 time). Defaults to 4.

    Returns:
        tuple: A tuple containing:
            - list[pretty_midi.Note]: List of notes from the MIDI file.
            - list[pretty_midi.Note]: List of notes representing the click track.
            - float: Total duration of the track in seconds based on target BPM.
            - float: Time offset (in seconds) for the start of the measure
                     immediately following the last event.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MIDI file not found: {filepath}")
    if meter <= 0:
        raise ValueError("Meter (beats per measure) must be positive.")
    if target_bpm <= 0:
        print("Warning: Target BPM is zero or negative. Click track and time offset calculation will be skipped.")
        # Allow processing notes, but click/offset is meaningless
        # target_bpm = 0 # Or raise error? For now, proceed but skip calcs.

    notes_list = []
    click_track_list = []
    active_notes = {}  # Store start times (in target ticks) of active notes: {(channel, pitch): start_tick}
    max_time_sec = 0.0 # Track the timestamp of the *end* of the last note or event

    try:
        mid = mido.MidiFile(filepath)
    except Exception as e:
        raise IOError(f"Could not read MIDI file {filepath}: {e}")

    original_ppqn = mid.ticks_per_beat
    if not original_ppqn or original_ppqn <= 0:
        print(f"Warning: MIDI file '{filepath}' has invalid ticks_per_beat ({original_ppqn}). Assuming 96.")
        original_ppqn = 96

    ppqn_ratio = float(target_ppqn) / original_ppqn if original_ppqn else 0 # Avoid division by zero
    target_tempo_usec = mido.bpm2tempo(target_bpm) if target_bpm > 0 else 0

    print(f"Processing '{filepath}'...")
    print(f"Original PPQN: {original_ppqn}, Target PPQN: {target_ppqn}, Target BPM: {target_bpm}, Meter: {meter}/4") # Assuming /4 for print
    print(f"PPQN Ratio: {ppqn_ratio:.4f}, Target Tempo (usec/beat): {target_tempo_usec}")

    current_tempo_usec = 500000 # Default MIDI tempo (120 BPM) if none found early
    # Find initial tempo if set in track 0
    if mid.type == 1 and len(mid.tracks) > 0:
         for msg in mid.tracks[0]:
             if msg.type == 'set_tempo':
                 current_tempo_usec = msg.tempo
                 print(f"Found initial tempo in track 0: {mido.tempo2bpm(current_tempo_usec):.2f} BPM")
                 break
    # If no set_tempo found, calculate tempo from target_bpm for conversion
    if current_tempo_usec == 500000 and target_bpm > 0:
         current_tempo_usec = target_tempo_usec # Use target tempo if no initial tempo found

    for i, track in enumerate(mid.tracks):
        print(f"--- Processing Track {i} ---")
        current_time_original_ticks = 0

        for msg in track:
            current_time_original_ticks += msg.time

            # Handle tempo changes within the track if necessary (more complex)
            # For this version, we assume a constant target tempo for conversion
            # if msg.type == 'set_tempo':
            #     current_tempo_usec = msg.tempo # Update tempo if it changes mid-track

            # Rescale time to target ticks
            current_time_target_ticks = round(current_time_original_ticks * ppqn_ratio) if ppqn_ratio else 0

            # Convert target ticks to seconds using target PPQN and TARGET BPM/Tempo
            current_time_sec = 0.0
            if target_ppqn > 0 and target_tempo_usec > 0:
                 current_time_sec = mido.tick2second(current_time_target_ticks, target_ppqn, target_tempo_usec)

            note_key = None
            is_note_on = False
            is_note_off = False
            pitch = -1
            velocity = 0

            if msg.type == 'note_on' and msg.velocity > 0:
                is_note_on = True
                pitch = msg.note
                velocity = msg.velocity
                note_key = (msg.channel, pitch)
                max_time_sec = max(max_time_sec, current_time_sec) # Note start updates max time

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                is_note_off = True
                pitch = msg.note
                note_key = (msg.channel, pitch)
                # Note off time is the crucial one for overall duration
                max_time_sec = max(max_time_sec, current_time_sec)

            elif msg.is_meta: # Other meta messages might indicate end of track
                 max_time_sec = max(max_time_sec, current_time_sec)


            # --- Handle Note On/Off Logic (modified slightly for clarity) ---
            if is_note_on:
                if note_key in active_notes:
                    # Overlap: Close previous note at the start of the new one
                    start_tick_prev, start_sec_prev, vel_prev = active_notes.pop(note_key)
                    end_sec_prev = current_time_sec
                    if end_sec_prev > start_sec_prev:
                        notes_list.append(pretty_midi.Note(velocity=vel_prev, pitch=pitch, start=start_sec_prev, end=end_sec_prev))
                        # print(f"Warning: Note On overlap {note_key} at {current_time_sec:.3f}s. Closing previous.")

                # Store note on info: start tick, start sec, velocity
                active_notes[note_key] = (current_time_target_ticks, current_time_sec, velocity)

            elif is_note_off:
                if note_key in active_notes:
                    start_tick, start_sec, vel = active_notes.pop(note_key)
                    end_sec = current_time_sec

                    if end_sec > start_sec:
                        pm_note = pretty_midi.Note(
                            velocity=vel, # Use stored velocity from note_on
                            pitch=pitch,
                            start=start_sec,
                            end=end_sec
                        )
                        notes_list.append(pm_note)
                    # else: Warn about zero duration if needed
                # else: Warn about note off for inactive note if needed

        # After processing a track, clear any remaining active notes (notes held until end)
        # Use the final max_time_sec as their end time
        keys_to_clear = list(active_notes.keys())
        for note_key in keys_to_clear:
             pitch = note_key[1] # Get pitch from key
             start_tick, start_sec, vel = active_notes.pop(note_key)
             end_sec = max_time_sec # End note at the very end of the track content
             if end_sec > start_sec:
                 print(f"Note {note_key} was still active at end of track. Closing at {max_time_sec:.3f}s.")
                 notes_list.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start_sec, end=end_sec))


    # --- Calculate Time Offset for Next Measure ---
    time_offset = 0.0
    if target_bpm > 0 and meter > 0:
        seconds_per_beat = 60.0 / target_bpm
        seconds_per_measure = seconds_per_beat * meter

        if seconds_per_measure > 0:
            # Find the end time of the measure containing the last event
            # Ceiling division gives the index of the measure *after* the last event
            measure_index_after_last = math.ceil(max_time_sec / seconds_per_measure)

            # The offset is the start time of that next measure
            time_offset = measure_index_after_last * seconds_per_measure

            # Handle tiny floating point inaccuracies near measure boundaries
            # If max_time_sec is very close to a measure boundary, ceiling might push it
            # unnecessarily. Let's add a small epsilon check.
            epsilon = 1e-9
            measure_boundary_time = (measure_index_after_last -1) * seconds_per_measure
            if max_time_sec > measure_boundary_time - epsilon and max_time_sec <= measure_boundary_time + epsilon:
                 # If max_time_sec is essentially AT the previous measure boundary, recalculate
                 measure_index_containing_last = math.floor(max_time_sec / seconds_per_measure)
                 time_offset = (measure_index_containing_last + 1) * seconds_per_measure


    # --- Generate Click Track ---
    if target_bpm > 0 and meter > 0:
        seconds_per_beat = 60.0 / target_bpm
        # Calculate number of beats up to the END of the measure containing the last note
        # This ensures the click track covers the full final measure.
        num_beats_total = math.ceil(time_offset / seconds_per_beat) if seconds_per_beat > 0 else 0

        print(f"\nEffective duration for clicks/offset: {time_offset:.2f} seconds")
        print(f"Generating click track ({num_beats_total} beats at {target_bpm} BPM, meter={meter})...")

        for i in range(num_beats_total): # Iterate beat by beat
            click_start_time = i * seconds_per_beat
            click_end_time = click_start_time + 0.05

            # Ensure click doesn't exceed the calculated offset time (start of next measure)
            # Although unlikely with short duration, it's good practice.
            click_end_time = min(click_end_time, time_offset)

            if click_end_time > click_start_time:
                # Determine pitch based on position in measure
                if i % meter == 0:
                    pitch = 75 # Beat 1
                else:
                    pitch = 56    # Other beats

                click_note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=click_start_time,
                    end=click_end_time
                )
                click_track_list.append(click_note)

    return notes_list, click_track_list, time_offset

def transport(send_pipe, recv_pipe):
    with open('./cfg.json', 'r') as f:
        cfg = json.load(f)
    MIDI_IN = cfg['MIDI_IN']
    CLOCK_IN = cfg['CLOCK_IN']
    MIDI_OUT = cfg['MIDI_OUT']
    PPQN = cfg['PPQN']
    METER = cfg['METER']
    MAX_BARS = cfg['MAX_BARS']
    VERBOSE = cfg['VERBOSE']
    BPM = cfg['BPM']
    COMM_CHANNEL = cfg['robot_settings']['COMM_CHANNEL']

    CLOCK_OUT_CHAN = cfg['CLOCK_OUT_CHAN']
    AI_OUT_CHAN = cfg['AI_OUT_CHAN']
    MUSICIAN_OUT_CHAN = cfg['MUSICIAN_OUT_CHAN']

    start_time = time.time()
    tick_start = time.time()
    tick = avg_delta = beat = 0
    active_notes = {}
    active_ai_notes = {}
    bars_played_user = 0
    bars_played_ai = -1
    send_to_ai = False
    notes_to_play = []
    sort = False
    bpm = 120
    ai_turn = False
    toggle_response = False

    with mido.open_input(CLOCK_IN) as clock_port, mido.open_input(MIDI_IN) as inport, mido.open_output(MIDI_OUT) as outport, mido.open_output(CLOCK_IN) as clock_control:
        clock_control.send(
            mido.Message(
                type = 'control_change', 
                control = 127,
                channel = COMM_CHANNEL,
                value = map_value(BPM),
                time = 0
            )
        )
        
        while True:
            if recv_pipe.poll(0.001):
                res = recv_pipe.recv()
                if isinstance(res, str):
                    if res == 'Done Loading':
                        clock_control.send(
                            mido.Message(
                                type = 'control_change', 
                                control = 126,
                                channel = COMM_CHANNEL,
                                value = 127,
                                time = 0
                            )
                        )

                        clock_control.send(
                            mido.Message(
                                type = 'control_change', 
                                control = 126,
                                channel = COMM_CHANNEL,
                                value = 126,
                                time = 0
                            )
                        )

                        clock_control.send(
                            mido.Message(
                                type = 'control_change', 
                                control = 123,
                                channel = COMM_CHANNEL,
                                value = 124,
                                time = 0
                            )
                        )
                        break
            time.sleep(0.001)

        print('starting')
        while True:
            for port in (clock_port, inport):
                for msg in port.iter_pending():
                    if recv_pipe.poll(0.001):
                        res = recv_pipe.recv()
                        if isinstance(res, str):
                            if res == 'done':
                                bars_played_user = 0
                        elif isinstance(res, tuple):
                            if ai_turn:
                                if res[0] == 'certainty':
                                    clock_control.send(
                                        mido.Message(
                                            type = 'control_change', 
                                            control = 125,
                                            channel = COMM_CHANNEL,
                                            value = map_value(res[1], 0.0, 1.0, 0, 127),
                                            time = 0
                                        )
                                    )
                                else:
                                    # Ping this on every AI note that is being received
                                    clock_control.send(
                                        mido.Message(
                                            type = 'control_change', 
                                            control = 126,
                                            channel = COMM_CHANNEL,
                                            value = 124,
                                            time = 0
                                        )
                                    )
                                    notes_to_play.append(res)
                                    sort = True

                    if sort:
                        notes_to_play.sort(key=lambda event: event[0])
                        sort = False

                    if msg.type =='control_change':
                        if msg.control == 22:
                            send_pipe.send(("temperature", map_value(msg.value, 0, 127, 0.1, 1.8, prec='float')))
                        # if msg.control == 21:
                        #     msg.channel = COMM_CHANNEL
                        #     clock_control.send(msg)

                    if msg.type == 'clock' and port.name == CLOCK_IN:
                        delta = time.time() - tick_start
                        avg_delta += delta

                        if (tick + 1) % PPQN == 0:
                            tick = 0
                            bpm = round(60 / avg_delta, 0)
                            avg_delta = 0

                            time_ = time.time() - start_time
                            if beat % METER == 0:
                                pitch = 75
                                if bars_played_user > 0:
                                    bars_played_user += 1
                                if bars_played_ai >= 0:
                                    bars_played_ai += 1
                            else:
                                pitch = 56

                            if beat % METER == 0 and beat != 0:
                                beat = 0
                            beat += 1

                            # Sending notes
                            if VERBOSE:
                                print('click', bars_played_ai, bars_played_user, toggle_response)
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

                    if bars_played_user == MAX_BARS + 1:
                        for note, (start, velocity) in active_notes.items():
                            msg = mido.Message('note_off', note=note, velocity=velocity, channel=MUSICIAN_OUT_CHAN)
                            outport.send(msg)
                            note = pretty_midi.Note(
                                velocity=velocity,
                                pitch=msg.note,
                                start=start,
                                end=(time.time() - start_time) 
                            )

                            send_pipe.send(('piano', note))
                        send_pipe.send('start_ai')
                        clock_control.send(
                            mido.Message(
                                type = 'control_change', 
                                control = 126,
                                channel = COMM_CHANNEL,
                                value = 125,
                                time = 0
                            )
                        )
                        bars_played_user = -1
                        bars_played_ai = 1
                        notes_to_play = []
                        ai_turn = True
                    
                    if bars_played_ai == MAX_BARS + 1:
                        send_pipe.send('stop_ai')
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
                        clock_control.send(
                            mido.Message(
                                type = 'control_change', 
                                control = 123,
                                channel = COMM_CHANNEL,
                                value = 124,
                                time = 0
                            )
                        )
                        ai_turn = False
                        bars_played_user = 0
                        bars_played_ai = -1
                        
                    if msg.type in ['note_on', 'note_off'] and bars_played_user >= 0 and bars_played_user <= MAX_BARS:
                        if msg.channel == COMM_CHANNEL and msg.type == 'note_on':
                            if msg.note == 0:
                                toggle_response = not toggle_response
                        if msg.type == 'note_on' and msg.velocity > 0:
                            msg.channel = MUSICIAN_OUT_CHAN
                            outport.send(msg)
                            active_notes[msg.note] = ((time.time() - start_time), msg.velocity)
                            if bars_played_user == 0:
                                bars_played_user += 1

                        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0) and msg.note in active_notes.keys():
                            msg.channel = MUSICIAN_OUT_CHAN
                            outport.send(msg)
                            start, velocity = active_notes.pop(msg.note)
                            note = pretty_midi.Note(
                                velocity=velocity,
                                pitch=msg.note,
                                start=start,
                                end=(time.time() - start_time) 
                            )

                            send_pipe.send(('piano', note))

                    if bars_played_ai >= 0:
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
          
            time.sleep(0.00001)

def AI_note_manager(send_transport, recv_transport, send_AI, receive_AI):
    with open('./cfg.json', 'r') as f:
        cfg = json.load(f)
    PPQN = cfg['PPQN']
    METER = cfg['METER']
    BPM = reverse_map_value(map_value(cfg['BPM']))
    CTX = cfg['AI_settings']['CTX']
    prompt_file = cfg['AI_settings']['prompt']
    if prompt_file != '':
        piano_notes, click_notes, time_offset = get_prompt('./ACGrand.mid', target_ppqn=PPQN, target_bpm=BPM, meter=METER)
    else:
        piano_notes = []
        click_notes = []

    ORG_OFFSET = 0.0
    get_next_offset = False
    played_notes = []
    while True:
        if receive_AI.poll(0.001):
            data = receive_AI.recv()
            if isinstance(data, str):
                if data == 'Done Loading':
                    send_transport.send('Done Loading')
            if isinstance(data, tuple):
                if data[0] == 'certainty':
                    send_transport.send(data)
            if isinstance(data, list):
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

                    pm = pretty_midi.PrettyMIDI(
                        resolution=PPQN,
                        initial_tempo=BPM
                    )
                    time_sig = pretty_midi.containers.TimeSignature(numerator=4, denominator=4, time=0)
                    pm.time_signature_changes.append(time_sig)

                    pm.instruments.append(piano)
                    pm.instruments.append(click_track)

                    send_AI.send(pm)
                elif data == 'stop_ai':
                    send_AI.send("stop")
                # send_transport.send('done')
            elif isinstance(data, tuple):
                if data[0] == 'click':
                    if get_next_offset:
                        ORG_OFFSET = data[1].start
                        get_next_offset = False
                    data[1].start += time_offset
                    data[1].end += time_offset
                    click_notes.append(data[1])
                elif data[0] == 'temperature':
                    send_AI.send(data)
                else:
                    data[1].start += time_offset
                    data[1].end += time_offset
                    piano_notes.append(data[1])
                    piano_notes = piano_notes[-CTX:] # Filter so the piano_notes at most get a ctx of -256
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

    send_pipe.send('Done Loading')

    while True:
        print('done')
        data = recv_pipe.recv()
        if isinstance(data, tuple):
            if data[0] == 'temperature':
                temperature = data[1]
            continue
        else:
            pm = data
            certainty = []


        print('start')
        if isinstance(pm, str):
            continue
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
            if recv_pipe.poll(0.001):
                if recv_pipe.recv() == "stop":
                    break

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

            certainty = calculate_certainty(logits)
            certainty.append(certainty.mean().item())  
            average_certainty = sum(certainty) / len(certainty)

            send_pipe.send(("certainty", average_certainty))

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
                print('Processed all bars done')
                send_pipe.send('done')
                break

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