import ffmpy
import glob
import sys
import subprocess
import tempfile
import soundfile as sf


def extract(file_like):
    try:
        stdout, stderr = process_file(file_like)
        amps, sample_rate = read_wave_from_bytes(stdout)
        return amps, sample_rate
    except ffmpy.FFRuntimeError as e:
        print('ffmpy.FFRuntimeError')
        print(e)
        return [], 0
    except:
        print('Unknown exception')
        return [], 0

def process_file(file_like):
    ff = ffmpy.FFmpeg(
        inputs={'pipe:0': None},
        outputs={'pipe:1': '-ar 44100 -acodec pcm_s16le -f wav'},
        global_options=['-y', '-loglevel panic']
    )
    stdout, stderr = ff.run(input_data=file_like.read(), stdout=subprocess.PIPE)
    return stdout, stderr

def read_wave_from_bytes(byte_string):
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(byte_string)
        amps, sample_rate = sf.read(temp.name)
        print(amps)
        temp.close()
        if amps.ndim is 2:
            amps = amps[:,0]
        else:
            amps = amps[0:]
        return amps, sample_rate