
import os
import math
import numpy as np
import librosa.beat
import librosa.feature
import scipy.signal
import librosa.display
import scipy.signal
import util


x, sr = util.load_audio('data/Pirates_of_the_Caribbean.wav')
chromagram = librosa.feature.chroma_stft(y=x, sr=sr, norm=2, hop_length=128, base_c = False )

chroma = np.argmax( chromagram, axis=0)
maxch = np.max(chromagram, axis=0)

record = []
current = None
count = 0
for i in range(len(chroma)):
    next_note = None
    if maxch[i] <= 0.8:
#        print(util.note[-1], end=' ')
        next_note = util.note[-1]
    else:
#        print(util.note[chroma[i]], end=' ')
        next_note = util.note[chroma[i]]
    if current == next_note or current is None:
        current = next_note
        count += 1
    else:
        record.append((current, count))
        current = next_note
        count = 1
record.append((current, count))

for a in record:
    print(a, end=' ')
