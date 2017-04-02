import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import pygame
from SimUI import *
import sys
import keras
import cv2

##save_data = True
save_data = False
# load X and Y data to concatenate to
X = np.load('X_1.npy')
Y = np.load('Y_1.npy')

RL = True
##RL = False
human_teach = True
##human_teach = False
rand_act_prob_init = .5
min_rand_act = 0.2
rand_act_reduc_rate = 0.1
r_count = 0
p_count = 0
R_reward_hist = []
S_reward_hist = []
wpt_hit_hist = []
gamma = 1.1 # applies a reward for very confident next steps


duration = 500 # seconds
epochs = 50 ## still use main duration

# weights display
weights_img = np.ones((150,100))
cv2.namedWindow('weights vis',cv2.WINDOW_NORMAL)
cv2.resizeWindow('weights vis',250,50) # images come in at 1920 x 1080 but that is inconvient for screen viewing

start_up_one_shot = True

try:
    new_weights = np.load('new_weights.npy')
    init_weights = np.load('init_weights.npy')
except FileNotFoundError: pass
try:
    if new_weights.all() == init_weights.all():
        print('weights havent changed')
##        sys.exit()
except NameError: pass

try:
    model = keras.models.load_model('sailbot_t1.h5')
    np.save('init_weights.npy', model.layers[1].get_weights()[0])
except OSError:
    print('file not found')
        
    
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

waypoint_rad = 100

# constants
rho_air = 1.225 # kg/m^3
rho_water = 1000 # kg/m^3

def knots_to_mps(knots):
    mps = knots*0.514
    return mps

wind_angle = np.deg2rad(90) # degrees to radians # defining ?? as upwind # where the wind is blowing FROM
wind_speed = knots_to_mps(7) # entered in knots, converted to meters / second

# parameters 
boat_mass = 5000 # kg
sail_area = 15 # square meters
h_foil_area = 1

starboard = False
port = False

## logs
bt_spd_abs = []
bt_abs_spd_dir = [] # not sure if this will be used
bt_pos_x = []
bt_pos_y = []

time_step = 1 # ms

def rot_center(image, angle):
    """rotate an image while keeping its center and size"""
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image

def get_boat_COG(boat_vel_x,boat_vel_y):# boat true heading
    global debug
    if(boat_vel_x>0.001 and boat_vel_y>0.001): # tolerances for overflow errs
        boat_abs_spd_dir = np.arctan(boat_vel_y/boat_vel_x)
##        print('quad A non axial')
        debug+= ' quad A '
    elif boat_vel_x<-0.001 and boat_vel_y>0.001:
        boat_abs_spd_dir = np.pi + np.arctan(boat_vel_y/boat_vel_x)
##        print('quad B non axial')
        debug += 'quad B'
    elif boat_vel_x<-0.001 and boat_vel_y<-0.001:
        boat_abs_spd_dir = np.deg2rad(180)+np.arctan(boat_vel_y/boat_vel_x)
##        print('quad C non axial')'
        debug += 'quad C'
    elif boat_vel_x>0.001 and boat_vel_y<-0.001:
        boat_abs_spd_dir = np.pi*2+np.arctan(boat_vel_y/boat_vel_x)
##        print('quad D non axial')
        debug += 'quad D'
    elif (boat_vel_y>0 and abs(boat_vel_y)>abs(boat_vel_x)):
        boat_abs_spd_dir = np.pi/2
##        print('moving + y')
    elif (boat_vel_y<0 and abs(boat_vel_y)>abs(boat_vel_x)):
        boat_abs_spd_dir = 3*np.pi/2
##        print('moving - y')
    elif(boat_vel_x>0 and abs(boat_vel_y)<abs(boat_vel_x)):
        boat_abs_spd_dir = 0
##        print('moving + x')
    elif(boat_vel_x<0 and abs(boat_vel_y)<abs(boat_vel_x)):
        boat_abs_spd_dir = np.pi
##        print('moving - x')
    else:
        boat_abs_spd_dir = 0
        print('not moving')
    return boat_abs_spd_dir

Autonomous = False
try:
    for epoch in np.arange(epochs):

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

        sail_trim_perc = 1.00 # percentage 0 - 100%, 100 = algned with boat heading, 0 = 90 degrees to boat heading

        waypoint = gen_waypt()
        pygame.draw.circle(gameDisplay,red,waypoint,waypoint_rad,0)

        t = 0 # time in milliseconds
        rand_act_prob = rand_act_prob_init
        
        while t < duration*1000/epochs and not crashed:
            if t == duration*500 and human_teach: human_teach = False
            
            RL_debug = ' ' 
            debug = ' '
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
            
            boat_abs_spd_dir = get_boat_COG(boat_vel_x,boat_vel_y)
            
        ##    debug += str(boat_vel_abs)+' '+str(np.rad2deg(boat_abs_spd_dir))
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
        ##    print('aoa:',np.rad2deg(aoa),'boat to wind:',np.rad2deg(wind_angle-boat_heading))
            ##  TODO CONSTRAIN SAIL EASE TO NOT EASE PAST HEAD TO WIND
            # following Cl Cd functions determined in octave plus gut/experience/theory
            air_stall_bluff = False
            luffing = False
            if aoa >= np.deg2rad(5) and aoa <= np.deg2rad(50):
                if aoa >=np.deg2rad(15):
                    Cl = 0.025*np.rad2deg(aoa)
                else:
                    Cl = 0.039*np.rad2deg(aoa)-0.193
                Cd = 0.001*np.square(np.rad2deg(aoa))
                status = 'lift mode'
        ##        print('lifting! aoa:',np.rad2deg(aoa))
            elif aoa < np.deg2rad(5):
                status = 'luffing/pinching/head to wind'
                Cl = 0
                Cd = .5
                luffing = True
        ##        print('luffing! aoa:',np.rad2deg(aoa))
            elif aoa > np.deg2rad(50):
                status = 'stalled/bluff body mode'
                air_stall_bluff = True
                Cl = 0
                Cd = 3  # TODO (optional) CREATE bluff body Cd curve to reward squareness to wind
        ##        print('stalled / bluff body! aoa:',np.rad2deg(aoa))
            else: print('weird error!')
            # accounting for lift orientation wrt global coordinates
            if starboard:
                if wind_angle >= 0 and wind_angle <= np.pi: Sail_L = -1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
                else: Sail_L = 1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
            elif port:
                if wind_angle >= 0 and wind_angle <= np.pi: Sail_L = 1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
                else: Sail_L = -1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
            Sail_D = 1/2*rho_air*sail_area*Cd*np.square(apparent_wind_speed)*1/10 # 1/10 fudge factor

            debug += 'aoa:'+str(np.rad2deg(aoa))
            # hydro foil lift and drag
            hydro_stall = False
            h_aoa = abs(boat_abs_spd_dir - boat_heading)
            if h_aoa > np.pi: h_aoa = np.pi*2 - h_aoa
        ##    print('haoa:',np.rad2deg(h_aoa))
            if h_aoa > np.deg2rad(0) and h_aoa < np.deg2rad(50):
                if h_aoa >= np.deg2rad(15): h_Cl = 0.025*np.rad2deg(h_aoa) + 0.35
                else: h_Cl = 0.048*np.rad2deg(h_aoa)
                h_Cd = 0.001*np.square(np.rad2deg(h_aoa))+0.1
            else:
                print('shit Hydro Stall, haoa:',np.rad2deg(h_aoa))
                hydro_stall = True
                h_Cl = 0
                h_Cd = 3
            if (boat_heading-boat_abs_spd_dir >= 0 and boat_heading-boat_abs_spd_dir <= np.deg2rad(180)) or boat_heading-boat_abs_spd_dir <= np.deg2rad(-180):
                h_lift_dir = boat_abs_spd_dir + np.deg2rad(90)
            else: h_lift_dir = boat_abs_spd_dir - np.deg2rad(90)
            debug += 'haoa:'+str(np.rad2deg(h_aoa))+'lift dir:'+str(np.rad2deg(h_lift_dir))
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
                + Sail_D*np.cos(wind_angle+np.deg2rad(180))                     # sail drag force
                +Sail_L*np.cos(wind_angle-np.pi/2) # major prob???             # sail life force
                - hydro_D*np.cos(boat_abs_spd_dir)                              # hydro foil drag force
                + hydro_L*np.cos(h_lift_dir)                                    # hydro foil lift force
                          )/boat_mass
            boat_acc_y = (
                boat_wind_drag_force*np.sin(wind_angle+np.deg2rad(180))         # wind drag
                - boat_visc_hull_drag*np.sin(boat_abs_spd_dir)                  # hull viscous drag
                + Sail_D*np.sin(wind_angle+np.deg2rad(180))                     # sail drag force
                + Sail_L*np.sin(wind_angle-np.pi/2) # major prob???             # sail life force
                - hydro_D*np.sin(boat_abs_spd_dir)                              # hydro foil drag force
                + hydro_L*np.sin(h_lift_dir)                                    # hydro foil lift force
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
                new_y = pix.pos[1] + boat_vel_abs*2*np.sin(boat_abs_spd_dir)
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
            wpt_hit = 0
            wpt_reward = 0
            if dist_to_waypoint < 100:
                waypoint = gen_waypt()
                wpt_reward = 10
                wpt_hit = 10
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
                try: # calc and tune reward in manual mode
                    qval_new = model.predict(newX)[0]
                    correction = qval
                    reward = 0
                    reward = VMG*3 + boat_vel_abs - 3*np.sin(h_aoa/2) + 3*gamma*np.argmax(qval_new[0:3])
                    # - 3*np.floor(np.mod(abs(course_to_waypoint - boat_abs_spd_dir),np.pi)/(np.pi/4))
                    if abs(course_to_waypoint-boat_heading) > np.pi: ar = np.pi*2 - abs(course_to_waypoint-boat_heading) 
                    else: ar = abs(course_to_waypoint-boat_heading)
                    reward -= ar*boat_vel_abs*1
                    RL_debug = str(ar)+ RL_debug
                    reward += np.cos(aoa-10)
                    if not hydro_stall:
                        reward = boat_vel_abs
                    else:
                        reward = -1/10*np.rad2deg(h_aoa)
                
        ##            correction[R_act] = reward
                    s_reward = gamma*np.argmax(qval_new[3:6])
                    if sail_trim_perc > .1: s_reward += np.cos(aoa-10)
        ##            correction[S_act] = reward
                    debug += ' reward:' + str(reward)
                except NameError: pass # first pass
                qval = model.predict(newX)[0]
            else: # reinforcement learning
                RL_debug = ' '
                # predict confidence of next move for reward
                qval_new = model.predict(newX)[0]
                # calculate reward 
                try:
                    correction = qval
                    reward = 0
                    S_reward = 0
                    R_reward = 0

                    if not hydro_stall:
                        reward += boat_vel_abs + 1
                    else: # hydro stalled 
                        reward += -1/20*np.rad2deg(h_aoa)
                        if R_act != 2 and R_act == last_Ract:
                            R_reward += 1 # reward consistency in getting out of irons
                        elif R_act == 2: R_reward -= 1
                        elif R_act != 2 and last_Ract != 2:
                            if Ract != last_Ract:
                                R_reward -= 1
##                        
                    reward += gamma*(np.argmax(qval_new[0:3])+np.argmax(qval_new[3:6]))/2
                    reward += VMG*3
##                    if air_stall_bluff:
##                        S_reward -= np.rad2deg(aoa)/80
##                    if sail_trim_perc < 1 and np.argmax(qval_new[3:6]) != 2 and luffing: S_reward -= 2

                    if boat_vel_abs/2 > VMG:
                        R_reward -= 2
                        if R_act != 2:
                            R_reward += 1
                            if R_act == last_Ract : R_reward += 1
                            else: R_reward -= 1

                    reward += wpt_reward # 10 if ran into waypoint
                    
                    R_reward += reward ##- 2
                    S_reward += reward #- 2
                        
                    correction[R_act] = R_reward
                    correction[S_act] = S_reward
                    
                    RL_debug += ' R reward:' + str(R_reward) + ' S reward:' + str(S_reward)
                    R_reward_hist.append(R_reward)
                    S_reward_hist.append(S_reward)
                    wpt_hit_hist.append(wpt_hit)
                except NameError: pass # first pass

                # reinforce model
                batchSize = 300
                try:
                    if p_count ==0 and r_count == 0:
                        if Xbatch.shape[0] >= batchSize: # wait to fit model till have a sufficient memory
                            Xbatch = np.roll(Xbatch,1,0)
                            Ybatch = np.roll(Ybatch,1,0)
                            Xbatch[0,:] = oldX
                            Ybatch[0,:] = np.reshape(correction,(1,6))
                            model.fit(Xbatch,Ybatch,batch_size = batchSize, nb_epoch = 1, verbose = 0) # should add custome call back
                        else: # build up model memory
                            Xbatch = np.concatenate((Xbatch,newX),0)
                            Ybatch = np.concatenate((Ybatch,np.reshape(correction,(1,6))),0)
                            RL_debug += ' building memory buffer'
                except NameError: # first pass
                    RL_debug += ' first pass '
                    Xbatch = newX
                    Ybatch = np.reshape(model.predict(newX)[0],(1,6))
                oldX = newX

                ## predict and act
                try:
                    last_Ract = R_act
                    last_Sact = S_act
                except NameError: pass # first pass
                
                qval = model.predict(newX)[0]
                keys = pygame.key.get_pressed()  #checking pressed keys
                if not human_teach and keys[pygame.K_s]: human_teach = True
                if human_teach : # human teaching
                    RL_debug += ' human action'
                    if keys[pygame.K_a]: human_teach = not human_teach
                    if keys[pygame.K_LEFT]: R_act = 0
                    elif keys[pygame.K_RIGHT]: R_act = 1
                    else: R_act = 2 # no action option
                    if keys[pygame.K_UP] and sail_trim_perc <= 1: S_act = 0
                    elif keys[pygame.K_DOWN] and sail_trim_perc >= 0: S_act = 1
                    else: S_act = 2 # no action option
                elif (np.random.rand(1)[0] < rand_act_prob and p_count == 0) or r_count > 0: # act randomly
                    if r_count > 0:
                        r_count -= 1
                        R_act = 2 # no action
                        S_act = 5 # no action
                    else:
                        r_count = 0
                        RL_debug +=' random action '
                        R_act = np.random.randint(0,3) # rudder
                        S_act = np.random.randint(0,3) # sail
                else: # choose predicted best action
        ##            print('pred:',qval)
                    if p_count > 0:
                        p_count -= 1
                        R_act = 2
                        S_act = 5
                    else:
                        p_count = 0
                        RL_debug += 'PREDICTING move'
                        R_act = np.argmax(qval[0:3])
                        S_act = np.argmax(qval[3:6])+3

                if R_act == 0: boat_heading += 1/360*2*np.pi
                elif R_act == 1: boat_heading -= 1/360*2*np.pi
                else: pass # no rudder action option
                if S_act == 0 and sail_trim_perc <= 1: sail_trim_perc += .08
                elif S_act == 1 and sail_trim_perc >= 0: sail_trim_perc -= .08
                else: pass # no trim action option

                if t == 0 or np.mod(t,10) == 0:
                    try: del weight_img
                    except NameError: pass
                    for layer in model.layers:
                        if len(layer.get_weights()) > 0:
                            sect = cv2.resize(layer.get_weights()[0],(50,50))
                            try:
                                weight_img = np.concatenate((weight_img,sect),1)
                            except NameError:
                                weight_img = sect
                    if abs(weight_img.min())>weight_img.max(): weight_img = weight_img/abs(weight_img.min())*255
                    else: weight_img = weight_img/weight_img.max()*255
                    cv2.imshow('weights vis',weight_img)
                    if start_up_one_shot:
                        start_up_one_shot = False
                        cv2.imwrite('initial weights.jpg',weight_img)
                        

                
                    
            boat_heading = np.mod(boat_heading,np.pi*2)

            if not Autonomous:
                try:
                    X = np.concatenate((X,newX),0)
                    Y = np.concatenate((Y,newY),0)
                except NameError:
                    X = newX
                    Y = newY

            if rand_act_prob > min_rand_act and np.mod(t,300) == 0 and r_count == 0 and p_count == 0:
                rand_act_prob -= rand_act_reduc_rate # increasingly rely on RL predictions

            RL_debug = ' e:'+str(epoch) + ' ' + RL_debug
            debug = ' e:'+str(epoch) + ' ' + debug
            if RL:
                RL_debug = 't:'+str(int(t/(duration*10)))+' ' + RL_debug
                print(RL_debug)
            else: print(debug)
except KeyboardInterrupt: pass
    
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

if RL:
    plt.figure(1)
    plt.ion()
    plt.plot(R_reward_hist, 'b',label='rudder reward')
    plt.plot(S_reward_hist, 'g',label='trim reward')
    plt.plot(wpt_hit_hist,'r',label='wpt_hit')
    ##plt.title('absolute speed')
    ##plt.xlabel('milliseconds')
    ##plt.ylabel('meters/sec')
    plt.legend()
    ##plt.axis('equal')
    plt.show()

pygame.quit()

if save_data:
    try:
        np.save('X_1.npy',X)
        np.save('Y_1.npy',Y)
    except NameError: pass


model.save('sailbot_t1.h5')
np.save('new_weights.npy', model.layers[1].get_weights()[0])
cv2.imwrite('end weights.jpg',weight_img)
