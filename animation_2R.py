import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the arm
class RoboticArm:
    def __init__(self, link_lengths):
        self.link_lengths = link_lengths
        self.joint_angles = [0, 0]  # Initial joint angles

    def forward_kinematics(self):
        theta1, theta2 = self.joint_angles
        x1 = self.link_lengths[0] * np.cos(theta1)
        y1 = self.link_lengths[0] * np.sin(theta1)
        x2 = x1 + self.link_lengths[1] * np.cos(theta1 + theta2)
        y2 = y1 + self.link_lengths[1] * np.sin(theta1 + theta2)
        return x1, y1, x2, y2

    def update_joints(self, angles):
        self.joint_angles = angles

# Create the figure for plotting
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

arm = RoboticArm([1, 1])
lines, = ax.plot([], [], 'o-', lw=2)  # Line for the arm

def init():
    lines.set_data([], [])
    return lines,

def animate(i):
    angles = [np.pi/6 * np.sin(i * np.pi / 50), np.pi/4 * np.cos(i * np.pi / 50)]
    arm.update_joints(angles)
    x1, y1, x2, y2 = arm.forward_kinematics()
    lines.set_data([0, x1, x2], [0, y1, y2])
    return lines,

anim = FuncAnimation(fig, animate, init_func=init, frames=100, interval=50, blit=True)
print(anim.)
# To display the animation in Jupyter notebook, use:
# from IPython.display import HTML
# HTML(anim.to_jshtml())

# To save the animation
# anim.save('robotic_arm.mp4', writer='ffmpeg')

# plt.show()
