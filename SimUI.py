import numpy as np
import pygame
pygame.init()

speed = 3 # m/s
direction = np.random.rand(1)[0]*np.pi*2

pixel_count = 25

display_width = 800
display_height = 600
size = [display_width, display_height]

black = (0,0,0)
white = (255,255,255)
red = (255, 0,0)
blue = (0,0,255)
light_blue = (103,202,235)

gameDisplay = pygame.display.set_mode(size) 

pygame.display.set_caption('Sailing Sim')

clock = pygame.time.Clock()

##headimg = pygame.image.load('Headshot_left_cam_Num1.png')

class OceanPix:
    def __init__(self):
        self.pos = [np.random.rand(1)[0]*display_width,np.random.rand(1)[0]*display_height]
    def Declare(self, name):
        print(name, "is a new person!")
    def NewPos(self,pos):
        self.pos = pos
    def EdgeOver(self,pos,direction):
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
                    self.pos = [0,int(np.random.rand(1)[0]*display_height)]
                else:
                    self.pos = [display_width,int(np.random.rand(1)[0]*display_height)]
            else:
                # create a top/bottom wall pixel
                if t2b:
                    self.pos = [int(np.random.rand(1)[0]*display_width),0]
                else:
                    self.pos = [int(np.random.rand(1)[0]*display_width),display_height]                    
      
Pixels = [ OceanPix() for i in range(pixel_count)]

        #############################
def piecepos(pos):
    pygame.draw.circle(gameDisplay,light_blue,[int(pos[0]),int(pos[1])],5)   
##    gameDisplay.blit(headimg,(x,y))

x = int((display_width * 0.45))
y = int((display_height * 0.8))

x_change = 0
y_change = 0

crashed = False

img_width = 75
img_height = 75
img = pygame.image.load('topview_b.png')
img = pygame.transform.scale(img,(img_width,img_height))
img = pygame.transform.rotate(img,90-np.rad2deg(direction))
##img = pygame.transform.rotate(img,325-90)

print(direction, np.rad2deg(direction))
while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: #pygame function that occurs when user hits X
            crashed = True
##        if event.type == pygame.KEYDOWN and event.key == pygame.K_x: #pygame function that occurs when user hits X
##            crashed = True
##        if event.type == pygame.KEYDOWN:
##            if event.key == pygame.K_LEFT:
##                direction += 1/360*2*np.pi
##    ##                x_change = -5
##                print('assss')
##            elif event.key == pygame.K_RIGHT:
##                direction -= 1/360*2*np.pi
##                print('sfd')
####                x_change = 5
##        if event.type == pygame.KEYUP:
##            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
##                x_change = 0
##    print(event)
##    x += x_change

            
    gameDisplay.fill(white)


    keys = pygame.key.get_pressed()  #checking pressed keys
    if keys[pygame.K_LEFT]:
        direction -= 1/360*2*np.pi
        
    if keys[pygame.K_RIGHT]:
        direction += 1/360*2*np.pi

    for pix in Pixels:
        piecepos(pix.pos)
        new_x = pix.pos[0] + speed*2*np.cos(direction)
        new_y = pix.pos[1] + speed*2*np.sin(direction)
        pix.NewPos([new_x,new_y])
        pix.EdgeOver(pix.pos,direction)
##    print(direction,np.cos(direction),np.sin(direction))

    img = pygame.image.load('topview_b.png')
    img = pygame.transform.scale(img,(img_width,img_height))
    img = pygame.transform.rotate(img,90-np.rad2deg(direction))
    gameDisplay.blit(img,(int(display_width/2-img_width/2),int(display_height/2-img_height/2)))
##    pygame.draw.polygon(gameDisplay,black,pointlist,0)        


        
    pygame.display.update()

    clock.tick(60)

pygame.quit()
##quit()
