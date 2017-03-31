import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import pygame
from SimUI import *
import sys
import keras

##save_data = True
save_data = False
# load X and Y data to concatenate to
X = np.load('X_1.npy')
Y = np.load('Y_1.npy')

##RL = True
RL = False
rand_act_prob = 1

model = keras.models.load_model('sailbot_t1.h5')
### UI
pygame.init()

speed = 0 # m/s
##direction = np.random.rand(1)[0]*np.pi*2
direction = 0

pixel_count = 25

display_width = 800
display_height = 600
size = [display_width, display_height]

sail_length = 60

black = (0,0,0)
white = (255,255,255)
red = (255, 0,0)
blue = (0,0,255)
light_blue = (103,202,235)
green = (0,255,0)

gameDisplay = pygame.display.set_mode(size) 

pygame.display.set_caption('Sailing Sim')

clock = pygame.time.Clock()

Pixels = [ OceanPix() for i in range(pixel_count)]

crashed = False

img_width = 100
img_height = 100
img = pygame.image.load('topview_b.png')
img = pygame.transform.scale(img,(img_width,img_height))
img = pygame.transform.rotate(img,90-np.rad2deg(direction))
### End UI

def gen_waypt():
    pos = (int((-.5+np.random.rand(1)[0])*2000+display_width/2),int((-.5+np.random.rand(1)[0])*2000+display_height/2))
    return pos
waypoint = gen_waypt()
waypoint_rad = 100
pygame.draw.circle(gameDisplay,red,waypoint,waypoint_rad,0)

# constants
rho_air = 1.225 # kg/m^3
rho_water = 1000 # kg/m^3

def knots_to_mps(knots):
    mps = knots*0.514
    return mps

wind_angle = np.deg2rad(90) # degrees to radians # defining ?? as upwind # where the wind is blowing FROM
wind_speed = knots_to_mps(7) # entered in knots, converted to meters / second

# parameters 
boat_mass = 500 # kg
sail_area = 15 # square meters
h_foil_area = 1

# initial conditions
boat_vel_x = 0
boat_vel_y = 0
boat_vel_abs = 0
boat_acc_x = 0
boat_acc_y = 0
boat_pos_x = 100
boat_pos_y = 100

boat_yaw = 0 # units? looking down, CCW positive
boat_heading = wind_angle+np.deg2rad(45)
boat_cog = 0 # needed?
boat_abs_spd_dir = 0

starboard = False
port = False
sail_trim_perc = 1.00 # percentage 0 - 100%, 100 = algned with boat heading, 0 = 90 degrees to boat heading

## logs
bt_spd_abs = []
bt_abs_spd_dir = [] # not sure if this will be used
bt_pos_x = []
bt_pos_y = []

t = 0 # time in milliseconds
time_step = 1 # ms
duration = 40 # seconds

def rot_center(image, angle):
    """rotate an image while keeping its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image

Autonomous = False
while t < duration*1000 and not crashed:
##        print('tic')
    # update position, uses velocities and accels stale by 1 time step
    boat_pos_x = boat_pos_x + boat_vel_x*time_step+1/2*boat_acc_x*np.square(time_step)
    boat_pos_y = boat_pos_y + boat_vel_y*time_step+1/2*boat_acc_y*np.square(time_step) # all positive typ. besides drawing fcns
    # log position updates
##    bt_pos_x.append(boat_pos_x)
##    bt_pos_y.append(boat_pos_y)
    
    # update velocity, uses stale accel by 1 time step
    boat_vel_x = boat_vel_x+boat_acc_x*time_step
    boat_vel_y = boat_vel_y+boat_acc_y*time_step
    boat_vel_abs = np.sqrt(np.square(boat_vel_x)+np.square(boat_vel_y))
    # boat true heading 
    if(boat_vel_x>0.001 and boat_vel_y>0.001): # tolerances for overflow errs
        boat_abs_spd_dir = np.arctan(boat_vel_y/boat_vel_x)
        print('quad A non axial')
    elif boat_vel_x<-0.001 and boat_vel_y>0.001:
        boat_abs_spd_dir = np.deg2rad(360)-np.arctan(boat_vel_y/boat_vel_x)
        print('quad B non axial')
    elif boat_vel_x<-0.001 and boat_vel_y<-0.001:
        boat_abs_spd_dir = np.deg2rad(180)+np.arctan(boat_vel_y/boat_vel_x)
        print('quad C non axial')
    elif boat_vel_x>0.001 and boat_vel_y<-0.001:
        boat_abs_spd_dir = np.deg2rad(360)-np.arctan(boat_vel_y/boat_vel_x)
        print('quad D non axial')
    elif (boat_vel_y>0 and abs(boat_vel_y)>abs(boat_vel_x)):
        boat_abs_spd_dir = np.pi/2
        print('moving + y')
    elif (boat_vel_y<0 and abs(boat_vel_y)>abs(boat_vel_x)):
        boat_abs_spd_dir = 3*np.pi/2
        print('moving - y')
    elif(boat_vel_x>0 and abs(boat_vel_y)<abs(boat_vel_x)):
        boat_abs_spd_dir = 0
        print('moving + x')
    elif(boat_vel_x<0 and abs(boat_vel_y)<abs(boat_vel_x)):
        boat_abs_spd_dir = np.pi
        print('moving - x')
    else:
        boat_abs_spd_dir = 0
        print('not moving')
    # log abs speed
##    bt_spd_abs.append(boat_vel_abs)
    
    ###### update acceleration with new forces
    # TODO CALC AND USE APPARENT WIND ANGLE
##    apparent_wind_speed = wind_speed + boat_vel_abs*np.cos(wind_angle-boat_abs_spd_dir)
    apparent_wind_speed = wind_speed # temporary
    
    if boat_heading >= wind_angle and boat_heading <= wind_angle + np.pi:
        starboard = True
        port = False
    else:
        starboard = False
        port = True
##    print('starboard:',starboard)
    # sail lift and drag
    if wind_angle - boat_heading > np.pi:
        aoa = abs(wind_angle-boat_heading)+np.pi-np.deg2rad(90)*(1-sail_trim_perc)
    elif wind_angle - boat_heading < -np.pi:
        aoa = np.pi*2 - abs(wind_angle-boat_heading)-np.deg2rad(90)*(1-sail_trim_perc) 
    else:
        aoa = abs(wind_angle-boat_heading)-np.deg2rad(90)*(1-sail_trim_perc)
##    if aoa > np.pi: aoa -= np.pi
    print('aoa:',np.rad2deg(aoa),'boat to wind:',np.rad2deg(wind_angle-boat_heading))
    ##  TODO CONSTRAIN SAIL EASE TO NOT EASE PAST HEAD TO WIND
    # following Cl Cd functions determined in octave plus gut/experience/theory
    if aoa >= np.deg2rad(5) and aoa <= np.deg2rad(50):
        if aoa >=np.deg2rad(15):
            Cl = 0.025*np.rad2deg(aoa)
        else:
            Cl = 0.039*np.rad2deg(aoa)-0.193
        Cd = 0.001*np.square(np.rad2deg(aoa))
        status = 'lift mode'
    elif aoa < np.deg2rad(5):
        status = 'luffing/pinching/head to wind'
        Cl = 0
        Cd = .5
    elif aoa > np.deg2rad(50):
        status = 'stalled/bluff body mode'
        Cl = 0
        Cd = 3  # TODO (optional) CREATE bluff body Cd curve to reward squareness to wind
    else: print('weird error!')
    # accounting for lift orientation wrt global coordinates
    if starboard:
        if wind_angle >= 0 and wind_angle <= np.pi: Sail_L = -1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
        else: Sail_L = 1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
    elif port:
        if wind_angle >= 0 and wind_angle <= np.pi: Sail_L = 1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
        else: Sail_L = -1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
    Sail_D = 1/2*rho_air*sail_area*Cd*np.square(apparent_wind_speed)*1/10 # 1/10 fudge factor

    # hydro foil lift and drag
    h_aoa = abs(boat_abs_spd_dir - boat_heading)
    if h_aoa > np.pi: h_aoa -= np.pi
##    print('haoa:',np.rad2deg(h_aoa))
    if h_aoa > np.deg2rad(0) and h_aoa < np.deg2rad(50):
        if h_aoa >= np.deg2rad(15): h_Cl = 0.025*np.rad2deg(h_aoa) + 0.35
        else: h_Cl = 0.048*np.rad2deg(h_aoa)
        h_Cd = 0.001*np.square(np.rad2deg(h_aoa))+0.1
    else:
        h_Cl = 0
        h_Cd = 3
    if boat_heading > boat_abs_spd_dir: h_lift_dir = boat_abs_spd_dir + np.deg2rad(90)
    else: h_lift_dir = boat_abs_spd_dir - np.deg2rad(90)
    hydro_L = 1/2*rho_water*h_foil_area*h_Cl*np.square(boat_vel_abs)/10 # 1/100 fudge factor
    hydro_D = 1/2*rho_water*h_foil_area*h_Cd*np.square(boat_vel_abs)/10 # 1/100 fudge factor
    status += ' '+str(hydro_L) + ' '+str(hydro_D)+' '+ str(Sail_L)+' '+str(Sail_D)
    # hull wind drag
    WAG_boat_Cd = .25
    boat_wind_drag_force = 1/2*WAG_boat_Cd*rho_air*apparent_wind_speed
    
    # hull viscous water drag
    WAG_Cv = 15
    boat_visc_hull_drag = WAG_Cv*np.square(boat_vel_abs)
    
    # acc = sumF/m
    boat_acc_x = (
        boat_wind_drag_force*np.cos(wind_angle+np.deg2rad(180))         # wind drag
        - boat_visc_hull_drag*np.cos(boat_abs_spd_dir)                  # hull viscous drag
##        + Sail_D*np.cos(wind_angle+np.deg2rad(180))                     # sail drag force
        +Sail_L*np.cos(wind_angle-np.pi/2) # major prob???             # sail life force
##        - hydro_D*np.cos(boat_abs_spd_dir)                              # hydro foil drag force
##        + hydro_L*np.cos(h_lift_dir)                                    # hydro foil lift force
                  )/boat_mass
    boat_acc_y = (
        boat_wind_drag_force*np.sin(wind_angle+np.deg2rad(180))         # wind drag
##        - boat_visc_hull_drag*np.sin(boat_abs_spd_dir)                  # hull viscous drag
##        + Sail_D*np.sin(wind_angle+np.deg2rad(180))                     # sail drag force
        + Sail_L*np.sin(wind_angle-np.pi/2) # major prob???             # sail life force
##        - hydro_D*np.sin(boat_abs_spd_dir)                              # hydro foil drag force
##        + hydro_L*np.sin(h_lift_dir)                                    # hydro foil lift force
        )/boat_mass
##    print(status)
    # update time
    t+=time_step
##    print(t)

##    print('tic')
    speed = boat_vel_abs
    direction = boat_abs_spd_dir
    for event in pygame.event.get():
        ### UI
        if event.type == pygame.QUIT: #pygame function that occurs when user hits X
            crashed = True
                
    gameDisplay.fill(white)


    for pix in Pixels:
        piecepos(gameDisplay,pix.pos)
        new_x = pix.pos[0] - boat_vel_abs*2*np.cos(boat_abs_spd_dir)
        new_y = pix.pos[1] - boat_vel_abs*2*np.sin(boat_abs_spd_dir)
        pix.NewPos([new_x,new_y])
        pix.EdgeOver(pix.pos,direction)
##    print(direction,np.cos(direction),np.sin(direction))

    img = pygame.image.load('topview_b.png')
    img = pygame.transform.scale(img,(img_width,img_height))
##    img = pygame.transform.rotate(img,np.rad2deg(boat_heading-np.pi/2))
    img = rot_center(img, np.rad2deg(boat_heading-np.pi/2)) # fucking rad function!
    gameDisplay.blit(img,(int(display_width/2-img_width/2),int(display_height/2-img_height/2)))
##    pygame.draw.circle(gameDisplay,red,(int(display_width/2),int(display_height/2)),5,0)
    sail_tack_pt = (int(display_width/2+sail_length/2*np.cos(boat_heading)),int(display_height/2-sail_length/2*np.sin(boat_heading)))
    if port:
        sail_clew_pt = (int(sail_tack_pt[0]-sail_length*np.cos(boat_heading+np.pi/2*(1-sail_trim_perc))),
                        int(sail_tack_pt[1]+sail_length*np.sin(boat_heading+np.pi/2*(1-sail_trim_perc))))
    elif starboard:
        sail_clew_pt = (int(sail_tack_pt[0]-sail_length*np.cos(boat_heading-np.pi/2*(1-sail_trim_perc))),
                        int(sail_tack_pt[1]+sail_length*np.sin(boat_heading-np.pi/2*(1-sail_trim_perc))))
    pygame.draw.line(gameDisplay,red,sail_tack_pt,sail_clew_pt,3)
    pygame.draw.line(gameDisplay,green,(int(display_width/2-img_width/2),int(display_height/2-img_height/2)),waypoint,3) # line to waypoint

    # update waypoint position
    waypoint_creep = [0,0]
    if int(boat_vel_abs*2*np.sin(boat_abs_spd_dir)) == 0:
        waypoint_creep[1] = boat_vel_abs*2*np.sin(boat_abs_spd_dir)
        wpt_y = int(waypoint[1] + boat_vel_abs*2*np.sin(boat_abs_spd_dir)+waypoint_creep[1])
    else:
        waypoint_creep[1] = 0
        wpt_y = int(waypoint[1] + boat_vel_abs*2*np.sin(boat_abs_spd_dir))
    if int(boat_vel_abs*2*np.cos(boat_abs_spd_dir)) == 0:
        waypoint_creep[0] = boat_vel_abs*2*np.cos(boat_abs_spd_dir)
        wpt_x = int(waypoint[0] - boat_vel_abs*2*np.cos(boat_abs_spd_dir) - waypoint_creep[0])
    else:
        waypoint_creep[0] = 0
        wpt_x = int(waypoint[0] - boat_vel_abs*2*np.cos(boat_abs_spd_dir))
    waypoint = (int(waypoint[0]- speed*2*np.cos(direction)),int(waypoint[1]+ speed*2*np.sin(direction)))
    pygame.draw.circle(gameDisplay,red,waypoint,waypoint_rad,0)
    # check hit waypoint
    dist_to_waypoint = np.sqrt(np.square((display_width/2-img_width/2)-waypoint[0])+np.square((display_width/2-img_width/2)-waypoint[1]))
    if dist_to_waypoint < 100:
        waypoint = gen_waypt()
##    print('dist to waypoint:',dist_to_waypoint)
    opp = (display_height/2-waypoint[1])
    adj = (waypoint[0]-display_width/2)
    if opp == 0 and adj < 0: course_to_waypoint = 0
    elif opp == 0 and adj > 0: course_to_waypoint = np.pi
    elif adj == 0 and opp > 0: course_to_waypoint = np.pi*3/2
    elif adj == 0 and opp < 0: course_to_waypoint = np.pi/2
    elif np.arctan(opp/adj) < 0 and adj > 0: course_to_waypoint = np.pi*2+np.arctan(opp/adj)
    elif np.arctan(opp/adj) < 0 and opp > 0: course_to_waypoint = np.pi+np.arctan(opp/adj)
    elif np.arctan(opp/adj) > 0 and opp < 0: course_to_waypoint = np.pi+np.arctan(opp/adj)
    else: course_to_waypoint = np.arctan(opp/adj)
##    print(' angle to waypoint:',np.rad2deg(course_to_waypoint))
    pygame.display.update()
    
    clock.tick(60)
    ### End UI
##        print(t)
    sys.stdout.flush()
    if t>duration*1000:
        print('time out')
        crashed = True

    try:
        boat_vel_abs_inputs = np.roll(boat_vel_abs_inputs,1)
        boat_vel_abs_inputs[0] = boat_vel_abs
        
        boat_abs_spd_dir_inputs = np.roll(boat_abs_spd_dir_inputs,1)
        boat_abs_spd_dir_inputs[0] = boat_abs_spd_dir

        boat_heading_inputs = np.roll(boat_heading_inputs,1)
        boat_heading_inputs[0] = boat_heading

        sail_trim_perc_inputs = np.roll(sail_trim_perc_inputs,1)
        sail_trim_perc_inputs[0] = sail_trim_perc

        aoa_inputs = np.roll(aoa_inputs,1)
        aoa_inputs[0] = aoa

        course_to_waypoint_inputs = np.roll(course_to_waypoint_inputs,1)
        course_to_waypoint_inputs[0] = course_to_waypoint
    except NameError:
        boat_vel_abs_inputs = np.ones((1,4))*boat_vel_abs
        boat_abs_spd_dir_inputs = np.ones((1,4))*boat_abs_spd_dir
        boat_heading_inputs = np.ones((1,4))*boat_heading
        sail_trim_perc_inputs = np.ones((1,4))*sail_trim_perc
        aoa_inputs = np.ones((1,4))*aoa
        course_to_waypoint_inputs = np.ones((1,4))*course_to_waypoint

    VMG = boat_vel_abs*np.cos(abs(course_to_waypoint-boat_abs_spd_dir))
##    print('VMG:',VMG,boat_vel_abs,'heading',(boat_heading),'course',boat_abs_spd_dir)
    newX = np.concatenate((boat_vel_abs_inputs,
                           boat_abs_spd_dir_inputs,
                           boat_heading_inputs,
                           sail_trim_perc_inputs,
                           aoa_inputs,
                           course_to_waypoint_inputs),1) # ignored distance to waypoint

##    print(np.reshape(newX,(4,6)))
##    print(model.predict(newX),Autonomous)

    newY = np.reshape(np.array([0,0,0,0,0,0]),(1,6))

    if not RL:
        keys = pygame.key.get_pressed()  #checking pressed keys
        if keys[pygame.K_a]: Autonomous = not Autonomous
        if not Autonomous:
            if keys[pygame.K_LEFT]:
                boat_heading += 1/360*2*np.pi
                newY[0,0] = 1
            elif keys[pygame.K_RIGHT]:
                boat_heading -= 1/360*2*np.pi
                newY[0,1] = 1
            else: newY[0,2] = 1 # no action option
            if keys[pygame.K_UP] and sail_trim_perc <= 1:
                sail_trim_perc += .01
                newY[0,3] = 1
            elif keys[pygame.K_DOWN] and sail_trim_perc >= 0:
                sail_trim_perc -= .01
                newY[0,4] = 1
            else: newY[0,5] = 1 # no action option
        else:
            dec_thresh = 0.5
            if model.predict(newX)[0][0] > dec_thresh: boat_heading += 1/360*2*np.pi
            elif model.predict(newX)[0][1] > dec_thresh: boat_heading -= 1/360*2*np.pi
            elif model.predict(newX)[0][2] > dec_thresh: pass # no rudder action option
            if model.predict(newX)[0][3] > dec_thresh: sail_trim_perc += .01
            elif model.predict(newX)[0][4] > dec_thresh: sail_trim_perc -= .01
            elif model.predict(newX)[0][5] > dec_thresh: pass # no trim action option
    else: # reinforcement learning
        qval = model.predict(newX)[0]
        if np.random.rand(1)[0] < rand_act_prob: # act randomly
            print('Random action')
            R_act = np.random.randint(0,3) # rudder
            S_act = np.random.randint(0,3) # sail
        else: # choose predicted best action
##            print('pred:',qval)
            R_act = np.argmax(qval[0:3])
            S_act = np.argmax(qval[3:6])+3
            
        if R_act == 0: boat_heading += 10/360*2*np.pi
        elif R_act == 1: boat_heading -= 10/360*2*np.pi
        else: pass # no rudder action option
        if S_act == 0 and sail_trim_perc <= 1: sail_trim_perc += .05
        elif S_act == 1 and sail_trim_perc >= 0: sail_trim_perc -= .05
        else: pass # no trim action option

        try:
            correction = qval
            if VMG -VMG_old > 0 or (VMG > 0 and VMG - VMG_old == 0): # reward   #  NEED TO ADD AN OR IF CONFIDENT
                correction[R_act] = VMG + 1
                correction[S_act] = VMG + 1
                print('reward')
            else: # punish
                correction[R_act] = VMG -1 
                correction[S_act] = VMG -1
                if rand_act_prob < .5: rand_act_prob += .05
                print('punish')
        except NameError: pass # VMG_old not defined yet
        VMG_old = VMG

        batchSize = 100
        try:
            if Xbatch.shape[0] >= batch_size:
                Xbatch = np.roll(Xbatch,1,0)
                Ybatch = np.roll(Ybatch,1,0)
                Xbatch[0,:] = newX
                Ybatch[0,:] = np.reshape(correction,(1,6))
                model.fit(Xbatch,Ybatch,batch_size = batchSize, nb_epoch = 1, verbose = 1)
            else:
                Xbatch = np.concatenate((Xbatch,newX),0)
                Ybatch = np.concatenate((Ybatch,np.reshape(correction,(1,6))),0)
        except NameError:
            Xbatch = newX
            Ybatch = np.reshape(correction,(1,6))
            
    boat_heading = np.mod(boat_heading,np.pi*2)

    if not Autonomous:
        try:
            X = np.concatenate((X,newX),0)
            Y = np.concatenate((Y,newY),0)
        except NameError:
            X = newX
            Y = newY

    if rand_act_prob > 0.1 and np.mod(t,300) == 0:
        rand_act_prob -= .1 # increasingly rely on RL predictions
        
##    gameDisplay.fill(white)
##    for pix in Pixels:
##        piecepos(gameDisplay,pix.pos)
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
####        clock.tick(60)
##    ### End UI
####        print(t)
##    sys.stdout.flush()
##    if t>duration*1000:
##        print('time out')
##        crashed = True

####### debugging prints #############
##    print("time =",t,"x =",boat_pos_x,"y=",boat_pos_y,"boat vel y",boat_vel_y)
##    print('accX:',boat_acc_x,'accY:',boat_acc_y)
##    print(boat_wind_drag_force, boat_visc_hull_drag)
##    print(boat_abs_spd_dir)
##    print(boat_visc_hull_drag*np.sin(-boat_abs_spd_dir))
##    print(apparent_wind_speed)
##    print(np.cos(-wind_angle-boat_abs_spd_dir))
##    print('X lift:',Sail_L*np.cos(wind_angle+np.pi/2),'Y lift:',Sail_L*np.sin(wind_angle+np.pi/2))
##    print('Lift:',Sail_L,'Drag:',Sail_D)
##if not crashed:
##    plt.figure(1)
##    plt.ion()
##    plt.plot(bt_spd_abs, 'bo',label='absolute bt spd')
##    ##plt.plot(history.val_loss, 'r',label='val loss')
##    plt.title('absolute speed')
##    plt.xlabel('milliseconds')
##    plt.ylabel('meters/sec')
##    plt.legend()
##    ##plt.axis('equal')
##    plt.show()
##
##    plt.figure(2)
##    ##axes = plt.gca()
##    ##axes.set_xlim([0,200])
##    ##axes.set_ylim([0,200])
##    plt.ion()
##    plt.plot(bt_pos_x,bt_pos_y, 'bo',label='boat position')
##    plt.legend()
##    plt.axis('equal')
##    plt.show()

pygame.quit()

if save_data:
    try:
        np.save('X_1.npy',X)
        np.save('Y_1.npy',Y)
    except NameError: pass


model.save('sailbot_t1.h5')
