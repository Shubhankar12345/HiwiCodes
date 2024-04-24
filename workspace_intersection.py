import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Finger kinematics

def finger_kinematics(ET,finger_angles):
    q_jts = finger_angles.shape
    finger_end_effector = np.zeros((q_jts[1],q_jts[0]))
    for i in range(q_jts[0]):
        q = q_joints[i]
        # Allocate the resulting forward kinematics array
        fk = np.eye(4)

        # Now we must loop over the ETs in the Panda
        for et in ET:
            if et.isjoint:
                # This ET is a variable joint
                # Use the q array to specify the joint angle for the variable ET
                fk = fk @ et.A(q[et.jindex])
            else:
                # This ET is static
                fk = fk @ et.A()
        finger_end_effector[:,i] = fk[0:3,3]
        
        return finger_end_effector 

# Index finger kinematic chain

# MCP joint (abduction-adduction)
IF_MCP_jrot_a = rtb.ET.Rz() # Variable rotation around the z axis
IF_MCP_joffset_a = rtb.ET.tz(-0.004)
IF_MCP_ltwist_a = rtb.ET.Rx(np.pi/2)
IF_MCP_llength_a = rtb.ET.tx(0.0155) # (5th Metacarpal length)
IF_MCP_ROM_a = np.linspace(-np.pi/4,0,300)

# MCP joint (flexion-extension)
IF_MCP_jrot_f = rtb.ET.Rz() # Variable rotation around the z axis
IF_MCP_llength_f = rtb.ET.tx(0.02) # (5th Metacarpal length)
IF_MCP_ROM_f = np.linspace(0,np.pi/2,300)

# PIP joint 
IF_PIP_jrot = rtb.ET.Rz() # Variable rotation around the z axis
IF_PIP_llength = rtb.ET.tx(0.025) # (5th Metacarpal length)
IF_PIP_ROM = np.linspace(0,np.pi/2,300)

# DIP joint
IF_DIP_jrot = rtb.ET.Rz() # Variable rotation around the z axis
IF_DIP_llength = rtb.ET.tx(0.036) # (5th Metacarpal length)
IF_DIP_ROM = np.linspace(0,np.pi/2,300)

IF_jspace = np.array([IF_MCP_ROM_a,IF_MCP_ROM_f,IF_MCP_])
# Little finger transformation matrix (wrist to fingertip)
IF = IF_MCP_jrot_a * IF_MCP_joffset_a * IF_MCP_ltwist_a * IF_MCP_llength_a * IF_MCP_jrot_f * IF_MCP_llength_f * IF_PIP_jrot * IF_PIP_llength * IF_DIP_jrot * IF_DIP_llength

# Thumb kinematic chain

# MCP joint (abduction-adduction)
TH_MCP_jrot_a = rtb.ET.Rz() # Variable rotation around the z axis
TH_MCP_joffset_a = rtb.ET.tz(-0.031)
TH_MCP_ltwist_a = rtb.ET.Rx(-np.pi/2)
TH_MCP_llength_a = rtb.ET.tx(0.021) # (5th Metacarpal length)
TH_MCP_ROM_a = np.linspace(-np.pi/6,np.pi/2,300)

# MCP joint (flexion-extension)
TH_MCP_jrot_f = rtb.ET.Rz() # Variable rotation around the z axis
TH_MCP_llength_f = rtb.ET.tx(0.02) # (5th Metacarpal length)
TH_MCP_ROM_f = np.linspace(-np.pi/2,0,300)

# PIP joint 
TH_PIP_jrot = rtb.ET.Rz() # Variable rotation around the z axis
TH_PIP_llength = rtb.ET.tx(0.025) # (5th Metacarpal length)
TH_PIP_ROM = np.linspace(-np.pi/2,0,300)

# DIP joint
TH_DIP_jrot = rtb.ET.Rz() # Variable rotation around the z axis
TH_DIP_llength = rtb.ET.tx(0.036) # (5th Metacarpal length)
TH_DIP_ROM = np.linspace(-np.pi/2,0,300)

# Little finger transformation matrix (wrist to fingertip)
TH = TH_MCP_jrot_a * TH_MCP_joffset_a * TH_MCP_ltwist_a * TH_MCP_llength_a * TH_MCP_jrot_f * TH_MCP_llength_f * TH_PIP_jrot * TH_PIP_llength * TH_DIP_jrot * TH_DIP_llength

print(TH)
