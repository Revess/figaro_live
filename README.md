### Installation

```python -m venv venv``` (or use Conda)

```pip install -e .```

For the musenet model:
```git checkout musenet```

Executing:
```python main.py```  

Do edit the cfg.json beforehand!
```
{
    "MIDI_IN": "Launchkey Mini MK3 MIDI Port", 
    "MIDI_OUT": "IAC Driver Bus 1",
    "METER": 4,
    "MAX_BARS": 4,
    "MAX_NOTES": 256,
    "MAX_MEASURES": 2,
    "PPQN": 24,
    "VERBOSE": true,
    "TEMPERATURE": 1.2
}
```

Here set the MIDI_IN and MIDI_OUT devices as well as the max_bars.