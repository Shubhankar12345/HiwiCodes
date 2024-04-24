import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot
# Define the link parameters for the parallel robot
links = [
    rtb.PrismaticDH(a=0, alpha=np.pi/2),
    rtb.PrismaticDH(a=0, alpha=-np.pi/2),
    rtb.PrismaticDH(a=0, alpha=0),
]

# Create the parallel robot model
robot = rtb.DHRobot(links, name='Parallel Robot')

# Define a range of joint configurations
q_range = np.linspace(-1, 1, 100)

# Initialize lists to store end-effector positions
end_effector_positions = []

# Calculate forward kinematics for each joint configuration
for q in q_range:
    # Set the joint configuration for the parallel robot
    q_joints = np.array([q, q, q])  # For simplicity, all prismatic joints have the same configuration
    
    # Calculate the forward kinematics for the current joint configuration
    ee_pose = robot.fkine(q_joints)
    
    # Extract the end-effector position
    end_effector_positions.append(ee_pose.t)

# Convert the list of end-effector positions to a numpy array for easier manipulation
end_effector_positions = np.array(end_effector_positions)

# Plot the end-effector positions
plt.plot(q_range, end_effector_positions[:, 0], label='X Position')
plt.plot(q_range, end_effector_positions[:, 1], label='Y Position')
plt.plot(q_range, end_effector_positions[:, 2], label='Z Position')

# Set plot labels and legend
plt.xlabel('Joint Configuration')
plt.ylabel('End-Effector Position')
plt.title('Forward Kinematics of Parallel Robot')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
