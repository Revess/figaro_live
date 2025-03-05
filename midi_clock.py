import mido, time

with mido.open_input('Launchkey Mini MK3:Launchkey Mini MK3 Launchkey Mi 20:0') as inport:
    s = time.time()
    for message in inport:
        if 'clock' in message.type:
            print(message, time.time() - s)
            s = time.time()