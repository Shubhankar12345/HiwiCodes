import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import orientation_2_vectors

# Little finger kinematic chain

# Finger placement
LF_jrot = rtb.ET.Rz(np.pi/2) # Constant rotation around the z axis
LF_joffset = rtb.ET.tz(0.02956) # joint offset
LF_ltwist = rtb.ET.Rx(np.pi/2) # Constant rotation around the z axis

# CMC joint
LF_CMC_jrot = rtb.ET.Rz() # Variable rotation around the z axis
LF_CMC_llength = rtb.ET.tx(0.043) # (5th Metacarpal length)

# MCP joint
LF_MCP_jrot = rtb.ET.Rz() # Variable rotation around the z axis
LF_MCP_ltwist = rtb.ET.Rx(np.pi/18) # Inclination angle

# PIP joint 
LF_PIP_jrot = rtb.ET.Rz() # Variable rotation around the z axis
LF_PIP_llength = rtb.ET.tx(0.06295) # (Proximal and Medial Phalanx combined length)

# DIP joint
LF_DIP_jrot = rtb.ET.Rz() # Variable rotation around the z axis
LF_DIP_llength = rtb.ET.tx(0.018) # (Distal Phalanx length)

# Little finger transformation matrix (wrist to fingertip)
LF = LF_jrot * LF_joffset * LF_ltwist * LF_CMC_jrot * LF_CMC_llength * LF_MCP_jrot * LF_MCP_ltwist * LF_PIP_jrot * LF_PIP_llength * LF_DIP_jrot * LF_DIP_llength

# Thumb kinematic chain

# Thumb placement
TH_jrot = rtb.ET.Rz(-np.pi) # Constant rotation around the z axis
TH_joffset = rtb.ET.tz(0.01161) # joint offset
TH_ltwist = rtb.ET.Rx(-np.pi/2) # Constant rotation around the x axis
TH_llength = rtb.ET.tx(0.028) # Link length

# Thumb placement 2
TH_jrot_2 = rtb.ET.Rz(np.pi/2) # Constant rotation around the z axis
TH_joffset_2 = rtb.ET.tz(0.02) # joint offset
TH_ltwist_2 = rtb.ET.Rx(np.pi/2) # Constant rotation around the x axis

# CMC joint
TH_CMC_jrot = rtb.ET.Rz() # Variable rotation around the z axis
TH_CMC_ltwist = rtb.ET.Rx(np.pi/2) # Constant rotation around the x axis
TH_CMC_llength = rtb.ET.tx(0.048945) # Metacarpal length

# MCP joint
TH_MCP_jrot = rtb.ET.Rz() # Variable rotation around the z axis
TH_MCP_llength = rtb.ET.tx(0.03822) # Proximal phalanx length

# IP joint
TH_IP_jrot = rtb.ET.Rz() # Variable rotation around the z axis
TH_IP_llength = rtb.ET.tx(0.03081) # Distal phalanx length

# Thumb transformation matrix (Wrist to fingertip)
TH = TH_jrot * TH_joffset * TH_ltwist * TH_llength * TH_jrot_2 * TH_joffset_2 * TH_ltwist_2 * TH_CMC_jrot * TH_CMC_ltwist * TH_CMC_llength * TH_MCP_jrot * TH_MCP_llength * TH_IP_jrot * TH_IP_llength


# Thumb joints ROM
q_joints = np.array([np.linspace(-np.pi/18, (7*np.pi)/18, 100), np.linspace((-11*np.pi)/180, (55*np.pi)/180, 100), np.linspace((-13*np.pi)/180, (55*np.pi)/180, 100)]).T
q_jts = q_joints.shape
TH_fingertips = np.zeros((q_jts[1],q_jts[0]))
for i in range(q_jts[0]):
    q = q_joints[i]
    # Allocate the resulting forward kinematics array
    fk = np.eye(4)

    # Now we must loop over the ETs in the Panda
    for et in TH:
        if et.isjoint:
            # This ET is a variable joint
            # Use the q array to specify the joint angle for the variable ET
            fk = fk @ et.A(q[et.jindex])
        else:
            # This ET is static
            fk = fk @ et.A()
    TH_fingertips[:,i] = fk[0:3,3]

# Little finger joints ROM
q_joints1 = np.array([np.linspace(-np.pi/18, (np.pi)/18, 100), np.linspace((-20*np.pi)/180, (65*np.pi)/180, 100), np.linspace((-20*np.pi)/180, (65*np.pi)/180, 100), np.linspace((-20*np.pi)/180, (75*np.pi)/180, 100)]).T
q_jts1 = q_joints1.shape
LF_fingertips = np.zeros((3,q_jts1[0]))
for i in range(q_jts1[0]):
    q = q_joints1[i]
    # Allocate the resulting forward kinematics array
    fk = np.eye(4)

    # Now we must loop over the ETs in the Panda
    for et in LF:
        if et.isjoint:
            # This ET is a variable joint
            # Use the q array to specify the joint angle for the variable ET
            fk = fk @ et.A(q[et.jindex])
        else:
            # This ET is static
            fk = fk @ et.A()
    LF_fingertips[:,i] = fk[0:3,3]

# Create a new matplotlib figure and its axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the points
ax.scatter(TH_fingertips[0,:], TH_fingertips[1,:], TH_fingertips[2,:], label = "Thumb")
ax.scatter(LF_fingertips[0,:], LF_fingertips[1,:], LF_fingertips[2,:], label="Little Finger")
plt.legend(loc="upper left")
plt.show()