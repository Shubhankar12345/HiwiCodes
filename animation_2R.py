import numpy as np
import math
import roboticstoolbox as rtb
import spatialmath as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from functools import partial

def finger_kinematics(ET,finger_angles):
    
    # Allocate the resulting forward kinematics array
    fk = np.eye(4)
    finger_end_effector = np.zeros((3,1))
    joints_pos = np.zeros((3,ET.n))
    # Now we must loop over the ETs in the Panda
    for et in ET:
        if et.isjoint:
            # This ET is a variable joint
            # Use the q array to specify the joint angle for the variable ET
            fk = fk @ et.A(finger_angles[et.jindex])
            joints_pos[:,et.jindex] = fk[0:3,3] 
        else:
            # This ET is static
            fk = fk @ et.A()
        
    finger_end_effector = fk[0:3,3]
    
    return finger_end_effector,joints_pos


q = np.array([np.pi/4,np.pi/4])

# 2r 
j1_rot = rtb.ET.Rz() # Variable rotation around the z axis
l1_length = rtb.ET.tx(1) # (5th Metacarpal length)
# j1_ROM = np.linspace(-np.pi/4,0,N_sample)
 
j2_rot = rtb.ET.Rz() # Variable rotation around the z axis
l2_length = rtb.ET.tx(1) # (5th Metacarpal length)
# j2_ROM = np.linspace(-np.pi/4,0,N_sample)

twor_arm = j1_rot * l1_length * j2_rot * l2_length

# Define start and end joint angles for both joints
# These are assumed values for demonstration, adjust for your specific robot geometry
theta_start_q1, theta_end_q1 = np.pi/4, np.pi/3  # Ranges for joint 1
theta_start_q2, theta_end_q2 = np.pi/6, np.pi/4  # Ranges for joint 2

# Total duration of the motion
T = 5  # in seconds
time_steps = np.linspace(0, T, 100)  # Generate 100 time steps from 0 to T
twor_end_effector = np.zeros((3,time_steps.shape[0]))
joints_fk = np.zeros((3,twor_arm.n,time_steps.shape[0]))
joints_trajectory = np.array([[theta_start_q1 + (theta_end_q1 - theta_start_q1) * (
        10 * (t/T)**3 - 15 * (t/T)**4 + 6 * (t/T)**5) for t in time_steps],[theta_start_q2 + (theta_end_q2 - theta_start_q2) * (
        10 * (t/T)**3 - 15 * (t/T)**4 + 6 * (t/T)**5) for t in time_steps]])

# twor_end_effector, twor_joints_fk = finger_kinematics(twor_arm,q)

# print(twor_end_effector, twor_joints_fk)
for i in range(time_steps.shape[0]):
    twor_end_effector[:,i], joints_fk[:,:,i]= finger_kinematics(twor_arm,joints_trajectory[:,i])

# print(twor_end_effector)
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()
# plt.plot(joints_fk[0,1,:],joints_fk[1,1,:],'k--')
# plt.show()
line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def animate(i):
    thisx = [0, joints_fk[0,1,i], twor_end_effector[0,i]]
    thisy = [0, joints_fk[1,1,i], twor_end_effector[1,i]]

    history_x = twor_end_effector[0,:i]
    history_y = twor_end_effector[1,:i]

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*0.05))
    return line, trace, time_text

anim = FuncAnimation(fig, animate, len(time_steps), interval=0.05*1000, blit=True)
plt.show()