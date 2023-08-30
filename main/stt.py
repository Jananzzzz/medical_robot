import subprocess

# from user_query.wav to user_query.txt
from paddlespeech.cli.asr.infer import ASRExecutor
import time
import os

wav_dir = "/Users/janan/Chinese-medical-dialogue-data/main/wav/"

# check the wav_dir every 0.01s if there is a user_query.wav file
while True:
    if os.path.exists(wav_dir + "user_query.wav"):
        break
    time.sleep(0.01)

asr = ASRExecutor()
user_query = asr(audio_file="/Users/janan/Chinese-medical-dialogue-data/main/wav/user_query.wav")
# write the user_query to user_query.txt
with open("/Users/janan/Chinese-medical-dialogue-data/main/conversation/user_query.txt", "w") as f:
    f.write(user_query)
print("generate user_query.txt successfully.")


