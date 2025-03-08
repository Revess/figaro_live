import threading as td
from queue import Queue
import pretty_midi
import time, json, mido
import mido, torch
import pickle as pkl
from figaro.input_representation import remi2midi
from figaro.live_utils import load_model, get_features, sample

model, vae_module = load_model('./checkpoints/figaro-expert.ckpt', './checkpoints/vq-vae.ckpt')

transport_in_queue = Queue()
transport_out_AI_manager_queue = Queue()

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

def main():
    with open('./test.pkl', 'rb') as f:
        x = pkl.load(f)[0]
    decoded = model.vocab.decode(x)
    played_notes = []
    click = None
    clicks = 0
    rel_start = 0.0

    while True:
        if not transport_out_AI_manager_queue.empty():
            click = transport_out_AI_manager_queue.get()
            if click.pitch == 75:
                clicks += 1
            if clicks > 2 and click.pitch == 75:
                rel_start = click.start
                
        if click is not None:
            if clicks > 2:
                for i in range(len(decoded)):
                    pm = remi2midi(decoded[:i])
                    for instrument in pm.instruments:
                        for note in instrument.notes:
                            if not any(all([
                                        note.start == inst.start, 
                                        note.end == inst.end, 
                                        note.pitch == inst.pitch, 
                                        note.velocity == inst.velocity
                                    ]) for inst in played_notes):
                                # Offset for playback
                                transport_in_queue.put((note.start + rel_start, 'note_on', note))
                                transport_in_queue.put((note.end + rel_start, 'note_off', note))
                                played_notes.append(note)

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

                    transport_out_AI_manager_queue.put(note)

                tick += 1
                tick_start = time.time()

            if len(notes_to_play) > 0 and next_ai_note is None:
                next_ai_note = notes_to_play.pop(0)

            if next_ai_note is not None:
                # print(next_ai_note[0], time.time() - start_time)
                if next_ai_note[0] < (time.time() - start_time):
                    m_note = mido.Message(next_ai_note[1], note=note.pitch, velocity=note.velocity)
                    outport.send(m_note)
                    next_ai_note = None

if __name__ == "__main__":
    main_ = td.Thread(target = main)
    transport_ = td.Thread(target = transport)

    main_.start()
    transport_.start()

    main_.join()
    transport_.join()