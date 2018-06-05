
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
chromagram = librosa.feature.chroma_stft(y=x, sr=sr, norm=None, base_c = False )

chroma = np.argmax( chromagram, axis=0)
maxch = np.max(chromagram, axis=0)
for i in range(len(chroma)):
    if maxch[i] < 0.5:
        print(util.note[-1], end=' ')
    else:
        print(util.note[chroma[i]], end=' ')