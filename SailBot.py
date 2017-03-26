import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

# constants
rho_air = 1.225 # kg/m^3
rho_water = 1000 # kg/m^3

def knots_to_mps(knots):
    mps = knots*0.514
    return mps

wind_angle = np.deg2rad(90) # degrees to radians # defining ?? as upwind # where the wind is blowing FROM
wind_speed = knots_to_mps(5) # entered in knots, converted to meters / second

# parameters 
boat_mass = 500 # kg
sail_area = 15 # square meters

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
duration = .75 # seconds
while t<duration*1000: 

    # update position, uses velocities and accels stale by 1 time step
    boat_pos_x = boat_pos_x + boat_vel_x*time_step+1/2*boat_acc_x*np.square(time_step)
    boat_pos_y = boat_pos_y + boat_vel_y*time_step+1/2*boat_acc_y*np.square(time_step)
    # log position updates
    bt_pos_x.append(boat_pos_x)
    bt_pos_y.append(boat_pos_y)
    
    # update velocity, uses stale accel by 1 time step
    boat_vel_x = boat_vel_x+boat_acc_x*time_step
    boat_vel_y = boat_vel_y+boat_acc_y*time_step
    boat_vel_abs = np.sqrt(np.square(boat_vel_x)+np.square(boat_vel_y))
    # boat true heading 
    if(boat_vel_x>0.001 and boat_vel_y>0.001): # tolerances for overflow errs
        boat_abs_spd_dir = np.arctan(boat_vel_y/boat_vel_x)
##        print('non axial')
    if boat_vel_x<-0.001 and boat_vel_y<-0.001:
        boat_abs_spd_dir = np.deg2rad(180)+np.arctan(boat_vel_y/boat_vel_x)
##        print('non axial')
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
    # log abs speed
    bt_spd_abs.append(boat_vel_abs)
    
    ###### update acceleration with new forces
    # TODO CALC AND USE APPARENT WIND ANGLE
    apparent_wind_speed = wind_speed + boat_vel_abs*np.cos(wind_angle-boat_abs_spd_dir)
    # sail lift and drag
    aoa = abs(wind_angle-boat_heading)
    if aoa <= np.deg2rad(180):
        aoa = abs(wind_angle-boat_heading-np.pi/2*(1-sail_trim_perc))
    if aoa > np.deg2rad(180):
        aoa = np.deg2rad(360) - abs(wind_angle-boat_heading+np.pi/2*(1-sail_trim_perc))
    ##  TODO CONSTRAIN SAIL EASE TO NOT EASE PAST HEAD TO WIND
    # following Cl Cd functions determined in octave plus gut/experience/theory
    if aoa >= np.deg2rad(5) and aoa <= np.deg2rad(50):
        if aoa >=np.deg2rad(15):
            Cl = 0.025*np.rad2deg(aoa)
        else:
            Cl = 0.039*np.rad2deg(aoa)-0.193
        Cd = 0.001*np.square(np.rad2deg(aoa))
    elif aoa < np.deg2rad(5):
        print('head to wind')
        Cl = 0
        Cd = .5
    elif aoa > np.deg2rad(50):
        print('stalled/bluff body mode')
        Cl = 0
        Cd = 3  # TODO (optional) CREATE bluff body Cd curve to reward squareness to wind
    else: print('weird error!')
    # accounting for lift orientation wrt global coordinates
    if boat_heading >= wind_angle and boat_heading <= wind_angle + np.pi:
        starboard = True
        port = False
    else:
        starboard = False
        port = True
    if starboard:
        if wind_angle >= 0 and wind_angle <= np.pi: Sail_L = -1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
        else: Sail_L = 1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
    elif port:
        if wind_angle >= 0 and wind_angle <= np.pi: Sail_L = 1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
        else: Sail_L = -1/2*rho_air*sail_area*Cl*np.square(apparent_wind_speed)
    Sail_D = 1/2*rho_air*sail_area*Cd*np.square(apparent_wind_speed)
  
    # hull wind drag
    WAG_boat_Cd = 2.5
    boat_wind_drag_force = 1/2*WAG_boat_Cd*rho_air*apparent_wind_speed
    
    # hull viscous drag
    WAG_Cv = 5
    boat_visc_hull_drag = WAG_Cv*np.square(boat_vel_abs)
    
    # acc = sumF/m
    boat_acc_x = (
        boat_wind_drag_force*np.cos(wind_angle+np.deg2rad(180))         # wind drag
        - boat_visc_hull_drag*np.cos(boat_abs_spd_dir)                  # hull viscous drag
        + Sail_D*np.cos(wind_angle+np.deg2rad(180))                     # sail drag force
        - Sail_L*np.cos(wind_angle+np.pi/2) # major prob???             # sail life force
                  )/boat_mass
    boat_acc_y = (
        boat_wind_drag_force*np.sin(wind_angle+np.deg2rad(180))         # wind drag
        - boat_visc_hull_drag*np.sin(boat_abs_spd_dir)                  # hull viscous drag
        + Sail_D*np.sin(wind_angle+np.deg2rad(180))                     # sail drag force
        + Sail_L*np.sin(wind_angle+np.pi/2) # major prob???             # sail life force
        )/boat_mass
    
    # update time
    t+=time_step

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
    
plt.figure(1)
plt.ion()
plt.plot(bt_spd_abs, 'bo',label='absolute bt spd')
##plt.plot(history.val_loss, 'r',label='val loss')
plt.title('absolute speed')
plt.xlabel('milliseconds')
plt.ylabel('meters/sec')
plt.legend()
##plt.axis('equal')
plt.show()

plt.figure(2)
##axes = plt.gca()
##axes.set_xlim([0,200])
##axes.set_ylim([0,200])
plt.ion()
plt.plot(bt_pos_x,bt_pos_y, 'bo',label='boat position')
plt.legend()
##plt.axis('equal')
plt.show()
