import os
import math
import numpy as np
import soundfile as sf
import scipy.io.wavfile as wav
import librosa.beat
import librosa.feature
import scipy.signal
import librosa.display
import scipy.signal

data_dir = 'data/'

tone = [ #A based
#Major
[1,0,1,0,1,1,0,1,0,1,0,1], # A
[1,1,0,1,0,1,1,0,1,0,1,0], # A#
[0,1,1,0,1,0,1,1,0,1,0,1], # B
[1,0,1,1,0,1,0,1,1,0,1,0], # C
[0,1,0,1,1,0,1,0,1,1,0,1], # C#
[1,0,1,0,1,1,0,1,0,1,1,0], # D
[0,1,0,1,0,1,1,0,1,0,1,1], # D#
[1,0,1,0,1,0,1,1,0,1,0,1], # E
[1,1,0,1,0,1,0,1,1,0,1,0], # F
[0,1,1,0,1,0,1,0,1,1,0,1], # F#
[1,0,1,1,0,1,0,1,0,1,1,0], # G
[0,1,0,1,1,0,1,0,1,0,1,1], # G#
#minor
[1,0,1,1,0,1,0,1,1,0,1,0], # a
[0,1,0,1,1,0,1,0,1,1,0,1], # a#
[1,0,1,0,1,1,0,1,0,1,1,0], # b
[0,1,0,1,0,1,1,0,1,0,1,1], # c
[1,0,1,0,1,0,1,1,0,1,0,1], # c#
[1,1,0,1,0,1,0,1,1,0,1,0], # d
[0,1,1,0,1,0,1,0,1,1,0,1], # d#
[1,0,1,1,0,1,0,1,0,1,1,0], # e
[0,1,0,1,1,0,1,0,1,0,1,1], # f
[1,0,1,0,1,1,0,1,0,1,0,1], # f#
[1,1,0,1,0,1,1,0,1,0,1,0], # g
[0,1,1,0,1,0,1,1,0,1,0,1]  # g#
]

note = { -1:'-', 0 : 'A', 1:'A#', 2:'B', 3:'C', 4:'C#', 5:'D', 6:'D#', 7:'E', 8:'F', 9:'F#', 10:'G', 11:'G#' }

def load_audio(filename):
    # You may use the audio I/O packages 
    fs, x = wav.read(filename)
    if x.dtype != 'float32':
        x = np.float32(x/32767.)
    if x.ndim > 1: # I want to deal only with single-channel signal now
        x = np.mean(x, axis = 1)
    return x, fs