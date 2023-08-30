import pyaudio
import wave

def record_and_save_wav(filename, duration=5, sample_rate=16000, channels=1, chunk_size=1024):
    audio_format = pyaudio.paInt16
    audio = pyaudio.PyAudio()

    stream = audio.open(format=audio_format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk_size)

    print("Recording...")

    frames = []

    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Recording done.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {filename}")

if __name__ == "__main__":
    output_filename = "/Users/janan/Chinese-medical-dialogue-data/main/wav/user_query.wav"
    recording_duration = 5  # in seconds

    record_and_save_wav(output_filename, duration=recording_duration)
