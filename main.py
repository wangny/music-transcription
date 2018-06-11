#!/usr/local/bin/python3

import os
import sys
import math
import numpy as np
import librosa.beat
import librosa.feature
import librosa.core
import scipy.signal
import librosa.display
import scipy.signal
import util
import TabWrite

def main(filename):
    print("start transcipt", filename)
    x, sr = util.load_audio(util.data_dir+filename)
    # beats per section
    bps = util.beats_per_section(util.data_dir+filename)
    #
    tempo = librosa.beat.tempo(y=x, sr=sr)
    #
    chromagram = librosa.feature.chroma_stft(y=x, sr=sr, norm=2, hop_length=util.hop_length, base_c = False )
    chroma = np.argmax( chromagram, axis=0)
    maxch = np.max(chromagram, axis=0)
    #
    pitches, magnitudes = librosa.core.piptrack(y=x, sr=sr, S=None, n_fft=2048, hop_length=util.hop_length, fmin=50.0, fmax=2500.0, threshold=0.6)
    index = np.argmax(magnitudes, axis = 0)
    #
    record = []
    current = None
    count = 0
    for i in range(len(chroma)):
        next_note = None
        if maxch[i] <= 0.8 or pitches[index[i], i] <= 0:
            next_note = util.note[-1]
        else:
            next_note = librosa.core.hz_to_note(np.float16(pitches[index[i], i]))
        if current == next_note or current is None:
            current = next_note
            count += 1
        else:
            record.append((current, count))
            current = next_note
            count = 1
    record.append((current, count))

    fpb = util.frequency_per_beat(x, sr)

    adjusted_record = []
    for i in range(len(record)):
        (n, t) = record[i]
        note = (t/fpb)*4
        note = round(note)/4
        if note > 0 :
            if n == '-':
                adjusted_record.append((n,-1, note))
            else:
                adjusted_record.append((n[:-1],int(n[-1]),note))

    print("start output")
    #output
    TabWrite.WriteTab(filename.split('.')[0], adjusted_record, int(bps), tempo)

    print("done transcipt", filename)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('Pirates_of_the_Caribbean.wav')



