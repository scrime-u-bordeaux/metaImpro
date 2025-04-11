import pygame

def get_input():
    for event in pygame.event.get():

         if event.type == pygame.KEYDOWN:

               print("A key is pressed down")
get_input()