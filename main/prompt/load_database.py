import pygame
import time



def play_audio(filepath):
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

time.sleep(5)
play_audio("/Users/janan/Chinese-medical-dialogue-data/main/prompt/wav/load_database.wav")