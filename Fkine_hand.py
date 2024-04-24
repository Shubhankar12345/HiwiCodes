import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import orientation_2_vectors

# Little finger kinematic chain

# MCP joint
LF_MCP = rtb.ET.Rz() # Variable rotation around the z axis
# q = np.pi/3 # supply the joint coordinate value to compute the transform
# T_MCP_theta = LF_MCP.A(q) #  Returns the elementary transform as a numpy array 
# sm_T_MCP_theta = sm.SE3(T_MCP_theta) # Returns the transformation as a SE(3) object
LF_MCP_joffset = rtb.ET.tz(-0.2004) # joint offset
LF_MCP_llength = rtb.ET.tx(0.3874) # link length

# PIP joint
LF_PIP = rtb.ET.Rz() # Variable rotation around the z axis
LF_PIP_llength = rtb.ET.tx(0.1968) # link length

# DIP joint
LF_DIP = rtb.ET.Rz() # Variable rotation around the z axis
LF_DIP_llength = rtb.ET.tx(0.059) # link length

# Little finger fingertip
LF_fingertip_llength = rtb.ET.tx(0.1135) # link length 

LF = LF_MCP * LF_MCP_joffset * LF_MCP_llength * LF_PIP * LF_PIP_llength * LF_DIP * LF_DIP_llength * LF_fingertip_llength

# print the number of joints in the little finger kinematic chain
print(f"The panda has {LF.n} joints")

# print the number of ETs in the little finger kinematic chain
print(f"The panda has {LF.m} ETs")

# Thumb kinematic chain

# Using the above methodolgy, we can calculate the forward kinematics of our Panda model
# First, we must define the joint coordinates q, to calculate the forward kinematics at
q = np.array([0, -0.3, 0.1])

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
print(fk[0:3,3].reshape(3,1))
# Pretty print our resulting forward kinematics using an SE3 object
# print(sm.SE3(fk))

t = np.linspace(0, 2*np.pi, 100)  # Time vector
trajec_LF = np.array([np.sin(2*np.pi*0.2*t),np.sin(2*np.pi*0.2*t),np.sin(2*np.pi*0.2*t)]).T

LF.plot(trajec_LF)
