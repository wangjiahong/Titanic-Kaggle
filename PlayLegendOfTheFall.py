
import pygame
from pygame import mixer
import glob
from random import randint


musics = glob.glob("J:/MY MUSIC/nice mood/*.mp3")

n_musics = len(musics)




def playMusic():
    # Generate random number:
    random_number = randint(0, len(musics)- 1)


    # Get music name
    music_name = musics[random_number]

    # Load music
    mixer.init()
    mixer.music.load(music_name)

    mixer.music.play()


def stopMusic():
    mixer.music.stop()
    
    
playMusic()
stopMusic()

##









