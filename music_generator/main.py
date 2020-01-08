import numpy as np
from math import pi
import sounddevice as sd
import json
import sys
import soundfile as sf


def display_usage_message():
    print("music generator")
    print("args:")
    print("1) source file name")
    print("2) result file name")


def open_json_file(filename):
    with open(filename) as file:
        data = json.load(file)
    return data


def translate_time(note_type, note_time):
    divider = 1
    if note_type == "note":
        divider = 1
    elif note_type == "half":
        divider = 2
    elif note_type == "quarter":
        divider = 4
    elif note_type == "eight":
        divider = 8
    elif note_type == "sixteen":
        divider = 16
    return note_time/divider


def generate_array_from_data(data, Fs, note_time):
    x = np.arange(0, 0, 1/Fs)
    for element in data:
        frequency = element['freq']
        time = translate_time(element['note'], note_time)
        # time = time*Fs
        # time = int(time)
        n = np.arange(0, time, 1/Fs)
        x1 = np.sin(2*pi*frequency*n)
        x = np.concatenate((x, x1))
    return x


def generate_music(args, Fs):
    source_filename = args[0]
    data = open_json_file(source_filename)
    music_data = generate_array_from_data(data, Fs, 1.0)
    sd.play(music_data, Fs)
    sd.wait()
    # mydata = sd.rec(music_data, Fs, channels=2, blocking=True)
    result_filename = args[1]
    sf.write(result_filename, music_data, Fs)


if __name__ == "__main__":
    print("xd")
    args = sys.argv[1:]
    Fs = 44100
    generate_music(args, Fs)
