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
import madmom

data_dir = 'data/'

note = { -1:'-', 0 : 'A', 1:'A#', 2:'B', 3:'C', 4:'C#', 5:'D', 6:'D#', 7:'E', 8:'F', 9:'F#', 10:'G', 11:'G#' }

sbeat_templates = ['SWW', 'SWSW', 'SWSWW', 'SWWSW', 'SWWSWW', 'SWSWSWW', 'SWWSWSW']
wbeat_templates = ['WSW', 'WSWS', 'WSWSW', 'WSWWS', 'WSWWSW', 'WSWSWSW', 'WSWWSWS']

hop_length=128

def load_audio(filename):
    # You may use the audio I/O packages 
    fs, x = wav.read(filename)
    if x.dtype != 'float32':
        x = np.float32(x/32767.)
    if x.ndim > 1: # I want to deal only with single-channel signal now
        x = np.mean(x, axis = 1)
    return x, fs

def frequency_per_beat(x, sr):
    t, b = librosa.beat.beat_track(y=x, sr=sr, onset_envelope=None, hop_length=hop_length, start_bpm=120.0, tightness=100, trim=True, bpm=None, units='frames')
    t = math.ceil(t)
    n_frame = math.floor(sr / hop_length)  #一秒有幾個frame
    beat_length = 60 / t #拍子長度(sec)
    fpb = math.floor(n_frame*beat_length)
    return fpb

def continuity(arr):
    length = len(arr)
    delete_idx = []
    #print(arr)
    for i in range(length):
        if i==0 and arr[i]+1 != arr[i+1]:
            delete_idx.append(i)
        elif arr[i] != arr[i-1]+1 and arr[i] != arr[i+1]-1:
            delete_idx.append(i)
    new_arr = np.delete(arr, delete_idx)
    #print(new_arr)
    return new_arr

def strong_weak_beat(arr):
    length = arr.shape[0]
    beats, dbeats = [], []
    for i in range(length):
        if arr[i][0]>0.1:
            beats.append(i)
        if (arr[i][1]>0.1):
            dbeats.append(i)

    beats = continuity(beats)
    dbeats = continuity(dbeats)
    isin_ = np.isin(beats, dbeats)
    length = beats.shape[0]
    cur, sw_beat, exist = beats[0], '', isin_[0]
    for i in range(1, length):
        if beats[i]==cur+1:
            cur = beats[i]
            if exist==False and isin_[i]==True:
                exist = True
        else:
            if exist==True:
                sw_beat = sw_beat+'S'
            else:
                sw_beat = sw_beat+'W'
            cur, exist = beats[i], isin_[i]

    return sw_beat

def check_beats(start, sw_beat, templates):
    scores = []
    length = len(sw_beat)
    for i in templates:
        l, cur, score = len(i), start, 0.0
        while (cur+l)<=length:
            if sw_beat[cur:cur+l]==i:
                score = score+1
            cur = cur+l
        scores.append([l, score])
    #
    prev_b, prev_score, s = 0.0, 0.0, []
    for i in scores:
        if i[0]==3:
            prev_b, prev_score = i[0], i[1]
        elif i[0]==prev_b:
            prev_score = prev_score+i[1]
        else:
            s.append([prev_b, prev_score])
            prev_b, prev_score = i[0], i[1]
    s.append([prev_b, prev_score])
   
    return s


def beats_per_section(fname):
    proc = madmom.features.beats.RNNDownBeatProcessor()
    arr = proc(fname)

    sw_beat = strong_weak_beat(arr)

    # Predict 強起拍
    length = len(sw_beat)
    start = 0
    for i in range(length):
        if sw_beat[i]=='S':
            start = i
            break
    s_scores = check_beats(start, sw_beat, sbeat_templates)
    # Predict 弱起拍
    for i in range(length):
        if sw_beat[i]=='W':
            start = i
            break
    w_scores = check_beats(start, sw_beat, wbeat_templates)
    # average of 強起拍 & 弱起拍
    bps_scores = np.sum([s_scores, w_scores], axis=0)
    #print(bps_scores)
    max_idx = np.argmax(bps_scores, axis=0)
    #print(bps_scores[max_idx[1]][0]/2)
    return bps_scores[max_idx[1]][0]/2



