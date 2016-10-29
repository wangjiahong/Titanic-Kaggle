
import pygame
from pygame import mixer
import glob
from random import randint


musics = glob.glob("J:/MY MUSIC/nice mood/*.mp3")
n_musics = len(musics)


youCanTrustInMe = 'J:/MY MUSIC/nice mood\\youcan trust in me.mp3'
n_end_of_English_songs = musics.index(youCanTrustInMe)   #122th music


English_musics = musics[:n_end_of_English_songs+1]

def playMusic():
    # Generate random number:
    random_number = randint(0, len(English_musics)- 1)

    # Name of a random music
    music_name = English_musics[random_number]

    # Load the music
    mixer.init()
    mixer.music.load(music_name)

    mixer.music.play()


def stopMusic():
    mixer.music.stop()
    
    
playMusic()
stopMusic()











