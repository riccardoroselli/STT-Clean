"""audio.py - the microphone (open only while recording), resampling, and the start/stop sounds."""
import os
import subprocess

import numpy as np
import sounddevice as sd

SOUND_START = "/System/Library/Sounds/Tink.aiff"
SOUND_STOP = "/System/Library/Sounds/Pop.aiff"
SOUND_CANCEL = "/System/Library/Sounds/Funk.aiff"


def play(path: str) -> None:
    if os.path.exists(path):
        subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_mic(sr: int, heard) -> tuple:
    """Started stream on the current default mic, (stream, its rate); heard(samples) gets every
    50 ms block (on the audio thread). Opened per recording, not at launch: an always-open mic keeps
    the orange privacy dot on and AirPods in call quality, and a stream whose device went away
    (AirPods back in the case) stops without any error."""
    sd._terminate()           # PortAudio reads the device list once: re-read it for mics plugged in / removed since
    sd._initialize()

    def opened(rate: int):
        return sd.InputStream(samplerate=rate, channels=1, dtype="float32", blocksize=int(rate * 0.05),
                              callback=lambda indata, *_: heard(indata[:, 0].copy()))
    try:
        rate = sr
        stream = opened(rate)
    except sd.PortAudioError:  # the mic can't do the model rate: record at its own, resample()
        rate = int(sd.query_devices(kind="input")["default_samplerate"])
        stream = opened(rate)
    stream.start()
    return stream, rate


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or len(audio) == 0:
        return audio
    n_out = int(round(len(audio) * dst_sr / src_sr))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
