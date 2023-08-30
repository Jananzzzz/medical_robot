from paddlespeech.cli.tts.infer import TTSExecutor
import pygame
import os
import time

# check the answer.txt every 0.01s if there is a answer.txt file
while True:
    if os.path.exists("/Users/janan/Chinese-medical-dialogue-data/main/conversation/answer.txt"):
        break
    time.sleep(0.01)

with open("/Users/janan/Chinese-medical-dialogue-data/main/conversation/answer.txt", "r") as f:
    Answer = f.read()

def play_audio(filepath):
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

tts = TTSExecutor()
tts(text=Answer, output="/Users/janan/Chinese-medical-dialogue-data/main/wav/answer.wav")
print("generate answer.wav successfully.")
# play the answer.wav
play_audio("/Users/janan/Chinese-medical-dialogue-data/main/wav/answer.wav")
print("play answer.wav successfully.")
