import numpy as np
import pygame

display_width = 800
display_height = 600
black = (0,0,0)
white = (255,255,255)
red = (255, 0,0)
blue = (0,0,255)
light_blue = (103,202,235)
class OceanPix:
    def __init__(self):
        self.pos = [np.random.rand(1)[0]*display_width,np.random.rand(1)[0]*display_height]
    def Declare(self, name):
        print(name, "is a new person!")
    def NewPos(self,pos):
        self.pos = pos
    def EdgeOver(self,pos,direction):
##        direction = -direction 
        if self.pos[0] < 0 or self.pos[0] > display_width or self.pos[1] < 0 or self.pos[1] > display_width:
            if np.cos(direction)>0:
                r2f = True
                #right to left
            else:
                r2f = False
                # left to right
            if np.sin(direction)>0:
                t2b = True
                #top to bottom
            else:
                t2b = False
                #bottom to top

            
            var = np.random.rand(1)[0]
##            print(direction,np.cos(direction),var)
            if var >(abs(np.sin(direction))):
                # create a side wall pixel
                if r2f:
                    self.pos = [display_width,int(np.random.rand(1)[0]*display_height)]
                else:
                    self.pos = [0,int(np.random.rand(1)[0]*display_height)]
            else:
                # create a top/bottom wall pixel
                if t2b:
                    self.pos = [int(np.random.rand(1)[0]*display_width),0]
                else:
                    self.pos = [int(np.random.rand(1)[0]*display_width),display_height]                    
      
        #############################
def piecepos(surface,pos):
    pygame.draw.circle(surface,light_blue,[int(pos[0]),int(pos[1])],5)   
##    gameDisplay.blit(headimg,(x,y))



##print(direction, np.rad2deg(direction))
##while not crashed:
##    for event in pygame.event.get():
##        if event.type == pygame.QUIT: #pygame function that occurs when user hits X
##            crashed = True
##            
##    gameDisplay.fill(white)
##
##
##    keys = pygame.key.get_pressed()  #checking pressed keys
##    if keys[pygame.K_LEFT]:
##        direction -= 1/360*2*np.pi
##        
##    if keys[pygame.K_RIGHT]:
##        direction += 1/360*2*np.pi
##
##    for pix in Pixels:
##        piecepos(pix.pos)
##        new_x = pix.pos[0] + speed*2*np.cos(direction)
##        new_y = pix.pos[1] + speed*2*np.sin(direction)
##        pix.NewPos([new_x,new_y])
##        pix.EdgeOver(pix.pos,direction)
####    print(direction,np.cos(direction),np.sin(direction))
##
##    img = pygame.image.load('topview_b.png')
##    img = pygame.transform.scale(img,(img_width,img_height))
##    img = pygame.transform.rotate(img,90-np.rad2deg(direction))
##    gameDisplay.blit(img,(int(display_width/2-img_width/2),int(display_height/2-img_height/2)))
####    pygame.draw.polygon(gameDisplay,black,pointlist,0)        
##
##    pygame.display.update()
##
##    clock.tick(60)

##pygame.quit()
##quit()
