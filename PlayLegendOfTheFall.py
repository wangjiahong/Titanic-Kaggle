# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:16:05 2016

@author: Jiahong
"""
def playMusic():
    import pygame
    from pygame import mixer
    mixer.init()
    mixer.music.load("E:/Legends Of The Fall.mp3")
    mixer.music.play()
#mixer.music.stop()



def stopMusic():
    mixer.music.stop()
stopMusic()

