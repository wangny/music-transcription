from midiutil.MidiFile import MIDIFile

pitch_dict = {'-': 0, 'C': 1, 'C#': 2, 'D': 3, 'D#': 4, 'E': 5, 'F': 6, 'F#': 7, 'G': 8, 'G#': 9, 'A': 10, 'A#': 11, 'B': 12};

def WriteTab(Filename, notes, bps, tempo):
    # create your MIDI object
    mf = MIDIFile(1)     # only 1 track
    track = 0   # the only track

    time = 0    # start at the beginning
    mf.addTrackName(track, time, "Main Track")
    mf.addTempo(track, time, tempo)
    mf.addTimeSignature(track, time, bps, 2, 24)

    # add some notes
    for i in range(0, len(notes)):
        
        channel = 0
        volume = 100
        if(notes[i][0] != '-'):
            pitch = 12*(notes[i][1]+1)-1+pitch_dict[notes[i][0]]
            duration = notes[i][2]
            mf.addNote(track, channel, pitch, time, duration, volume)
        time = time + notes[i][2]

    # write it to disk
    with open(Filename + ".mid", 'wb') as outf:
        mf.writeFile(outf)