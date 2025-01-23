import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import DHRobot, RevoluteDH
import matplotlib.pyplot as plt
from spatialmath import SE3
from sklearn.metrics.pairwise import euclidean_distances 
from scipy.optimize import curve_fit
from hand_metrics import orientation_similarity_metric
import two_vec_orientation as tvo
import force_comp as fc

def compute_workspace(ET):
    workspace_pts = np.zeros((3,N_sample))
    workspace_pts_config = []
    for i in range(N_sample):
        if(ET.name=="Thumb"):
            finger_angles = np.array([[0],-np.radians(47)-np.random.choice(TH_CMC_ROM_a,1,replace=False),np.random.choice(TH_CMC_ROM_f,1,replace=False),np.random.choice(TH_MCP_ROM_a,1,replace=False),np.random.choice(TH_MCP_ROM_f,1,replace=False),np.random.choice(TH_IP_ROM,1,replace=False)])
        elif(ET.name=="LittleFinger"):
            finger_angles = np.array([[0],[0],np.random.choice(LF_CMC_ROM,1,replace=False),np.random.choice(LF_MCP_ROM_a,1,replace=False),np.random.choice(LF_MCP_ROM_f,1,replace=False),[0],np.random.choice(LF_PIP_ROM,1,replace=False),[0],np.random.choice(LF_DIP_ROM,1,replace=False)])     
        elif(ET.name=="LittleFingerProto2"):
            finger_angles = np.array([[0],[0],np.random.choice(LF_MCP_ROM_a,1,replace=False),np.random.choice(LF_MCP_ROM_f,1,replace=False),[0],np.random.choice(LF_PIP_ROM,1,replace=False),[0],np.random.choice(LF_DIP_ROM,1,replace=False)])     
        elif(ET.name=="IndexFinger"):
            finger_angles = np.array([[0],[0],np.random.choice(IF_MCP_ROM_a,1,replace=False),[0],np.random.choice(IF_MCP_ROM_f,1,replace=False),np.random.choice(IF_PIP_ROM,1,replace=False),np.random.choice(IF_DIP_ROM,1,replace=False)])        
        elif(ET.name=="MiddleFinger"):
            finger_angles = np.array([[0],np.random.choice(MF_MCP_ROM_a,1,replace=False),np.random.choice(MF_MCP_ROM_f,1,replace=False),np.random.choice(MF_PIP_ROM,1,replace=False),np.random.choice(MF_DIP_ROM,1,replace=False)])                      
        elif(ET.name=="RingFingerProto2"):
            finger_angles = np.array([[0],[0],np.random.choice(RF_MCP_ROM_a,1,replace=False),np.random.choice(RF_MCP_ROM_f,1,replace=False),[0],np.random.choice(RF_PIP_ROM,1,replace=False),[0],np.random.choice(RF_DIP_ROM,1,replace=False)])  
        else:
            finger_angles = np.array([[0],[0],np.random.choice(RF_CMC_ROM,1,replace=False),np.random.choice(RF_MCP_ROM_a,1,replace=False),np.random.choice(RF_MCP_ROM_f,1,replace=False),[0],np.random.choice(RF_PIP_ROM,1,replace=False),[0],np.random.choice(RF_DIP_ROM,1,replace=False)])  
        finger_angles = finger_angles.reshape(finger_angles.shape[0],)
        workspace_pts[:,i] = ET.fkine(finger_angles).A[0:3,3]
        workspace_pts_config.append(finger_angles)
    
    workspace_pts_config = np.array(workspace_pts_config).reshape((len(workspace_pts_config),len(workspace_pts_config[0])))
    return workspace_pts, workspace_pts_config

def project_points_to_axis(points, axis, axis_point):
    """Project points onto a given axis. Assumes axis is normalized."""
    points_centered = points - axis_point
    projections = np.dot(points_centered, axis)[:, None] * axis
    projected_points = projections + axis_point
    return projected_points, np.dot(points_centered, axis)

def filter_points_near_cylinder(workspace_points, workspace_angles, cylinder_points, cylinder_axis, cylinder_radius, cylinder_lowerlimit, cylinder_upper_limit, tolerance=1):
    """Filter points that are outside the cylinder and within a given tolerance from the cylinder's surface,
    and also within the vertical bounds of the cylinder from -0.1 to 0.1."""
    # Normalize the cylinder axis vector
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)
    
    # Calculate the axis point (base point of the cylinder)
    axis_point = np.mean(cylinder_points, axis=0)  # Typically, adjust this if needed to match the -0.1 base point
    
    # Project workspace points onto the cylinder's axis
    projected_points, projections = project_points_to_axis(workspace_points, cylinder_axis, axis_point)
    
    # Calculate perpendicular distances from workspace points to the axis
    distances_to_axis = np.linalg.norm(workspace_points - projected_points, axis=1)
    
    # Define the lower and upper limits for the radial distance
    min_distance = cylinder_radius  # Points must be outside the cylinder
    max_distance = cylinder_radius + tolerance / 1000.0  # Convert tolerance from mm to meters if necessary
    
    # Check axial bounds: projections must be within -0.1 to 0.1
    # axial_bounds_mask = (projections >= -cylinder_limit) & (projections <= cylinder_limit)
    axial_bounds_mask = (projections >= cylinder_lowerlimit) & (projections <= cylinder_upper_limit)
    
    # Filter points that are outside the cylinder but within the tolerance from its surface
    radial_mask = (distances_to_axis > min_distance) & (distances_to_axis <= max_distance)
    
    # Combine masks
    mask = axial_bounds_mask & radial_mask
    nearby_points = workspace_points[mask]
    nearby_points_angles = workspace_angles[mask]
    return nearby_points, nearby_points_angles

N_sample = 5000  # Number of samples
np.random.seed(0)

TH = DHRobot([
    RevoluteDH(a=0.028, alpha=0, d=0.02, offset=np.radians(153.5)),
    RevoluteDH(a=0.0, alpha= np.pi/2, d=0.0),
    RevoluteDH(a=0.048945, alpha= -np.radians(150), d=0.0),
    RevoluteDH(a=0.0085, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.03822, alpha=0.0, d=0.0),
    RevoluteDH(a=0.03081, alpha=0.0, d=0.0),
], name='Thumb')

TH_CMC_ROM_a = np.linspace(-np.radians(15),np.radians(60),N_sample)
TH_CMC_ROM_f = np.linspace(0,np.radians(60),N_sample)
TH_MCP_ROM_a = np.linspace(0,np.radians(30),N_sample)
TH_MCP_ROM_f = np.linspace(0,np.radians(55),N_sample)
TH_IP_ROM = np.linspace(0,np.radians(80),N_sample)


IF = DHRobot([
    RevoluteDH(a=0.0, alpha=0, d=0, offset=np.radians(106.5)),
    RevoluteDH(a=0.088, alpha=0, d=0, offset=0.0), 
    RevoluteDH(a=0.0085, alpha=0.0, d=0),
    RevoluteDH(a=0.0, alpha= np.pi/2, d=0, offset=-np.radians(10)),
    RevoluteDH(a=0.047775, alpha= 0, d=0),
    RevoluteDH(a=0.027885, alpha= 0, d=0),
    RevoluteDH(a=0.018915, alpha=0, d=0),
], name='IndexFinger')

IF_MCP_ROM_a = np.linspace(-np.pi/6,np.pi/6,N_sample)
IF_MCP_ROM_f = np.linspace(0,(4*np.pi)/9,N_sample)
IF_PIP_ROM = np.linspace(0,(5*np.pi)/9,N_sample)
IF_DIP_ROM = np.linspace(0,np.pi/2,N_sample)

MF = DHRobot([
    RevoluteDH(a=0.087, alpha=0.0, d=0.0, offset=np.pi/2),
    RevoluteDH(a=0.0085, alpha = np.pi/2, d=0.0),
    RevoluteDH(a=0.05187, alpha=0.0, d=0.0),
    RevoluteDH(a=0.03315, alpha=0.0, d=0.0),
    RevoluteDH(a=0.018915, alpha=0.0, d=0.0),
], name='MiddleFinger')

MF_MCP_ROM_a = np.linspace(-np.radians(22.5),np.radians(22.5),N_sample)
MF_MCP_ROM_f = np.linspace(0,np.radians(80),N_sample)
MF_PIP_ROM = np.linspace(0,np.radians(100),N_sample)
MF_DIP_ROM = np.linspace(0,np.pi/2,N_sample)

RF = DHRobot([
    RevoluteDH(a=0.0, alpha=0.0, d=0.0, offset=np.radians(72.5)),
    RevoluteDH(a=0.031, alpha=np.pi/2, d=0.0, offset=0.0),
    RevoluteDH(a=0.0485, alpha=np.pi/2, d=0.0),
    RevoluteDH(a=0.0085, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.04758, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0, offset=np.radians(7)),
    RevoluteDH(a=0.032175, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0,offset=np.radians(7)),
    RevoluteDH(a=0.02865, alpha=0.0, d=0.0),
], name='RingFinger')

RF_proto2 = DHRobot([
    RevoluteDH(a=0.0, alpha=0.0, d=0.0, offset=np.radians(72.5)),
    RevoluteDH(a=0.0795, alpha=np.pi, d=0.0, offset=0.0),
    RevoluteDH(a=0.0085, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.04758, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0, offset=np.radians(7)),
    RevoluteDH(a=0.032175, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0,offset=np.radians(7)),
    RevoluteDH(a=0.020865, alpha=0.0, d=0.0),
], name='RingFingerProto2')

RF_CMC_ROM = np.linspace(0,np.radians(10),N_sample)
RF_MCP_ROM_a = np.linspace(-np.radians(22.5),np.radians(22.5),N_sample)
RF_MCP_ROM_f = np.linspace(0,np.radians(80),N_sample)
RF_PIP_ROM = np.linspace(0,np.radians(100),N_sample)
RF_DIP_ROM = np.linspace(0,np.pi/2,N_sample)

LF = DHRobot([
    RevoluteDH(a=0.0, alpha=0.0, d=0.0, offset=np.radians(63.5)),
    RevoluteDH(a=0.033, alpha=np.pi/2, d=0.0, offset=0.0), 
    RevoluteDH(a=0.043, alpha= np.pi/2, d=0.0),
    RevoluteDH(a=0.0085, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.03978, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0, offset=np.radians(10)),
    RevoluteDH(a=0.022815, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0, offset=np.radians(10)),
    RevoluteDH(a=0.018135, alpha=0.0, d=0.0),
], name='LittleFinger')

LF_proto2 = DHRobot([
    RevoluteDH(a=0.0, alpha=0.0, d=0.0, offset=np.radians(63.5)),
    RevoluteDH(a=0.076, alpha=np.pi, d=0.0, offset=0.0),
    RevoluteDH(a=0.0085, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.03978, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0, offset=np.radians(10)),
    RevoluteDH(a=0.022815, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0,offset=np.radians(10)),
    RevoluteDH(a=0.018135, alpha=0.0, d=0.0),
], name='LittleFingerProto2')

LF_CMC_ROM = np.linspace(0,np.radians(20),N_sample)
LF_MCP_ROM_a = np.linspace(-np.radians(25),np.radians(25),N_sample)
LF_MCP_ROM_f = np.linspace(0,np.radians(80),N_sample)
LF_PIP_ROM = np.linspace(0,np.radians(100),N_sample)
LF_DIP_ROM = np.linspace(0,np.pi/2,N_sample)

thumb_wksp, thumb_jt_angles = compute_workspace(TH)
index_finger_wksp, index_finger_jt_angles = compute_workspace(IF)
middle_finger_wksp, middle_finger_jt_angles = compute_workspace(MF)
# ring_finger_wksp, ring_finger_jt_angles = compute_workspace(RF)
ring_finger_wksp_proto2, ring_finger_jt_angles_proto2 = compute_workspace(RF_proto2)
little_finger_wksp, little_finger_jt_angles = compute_workspace(LF)
# little_finger_wksp_proto2, little_finger_jt_angles_proto2 = compute_workspace(LF_proto2)

r = 0.02
theta = np.linspace(-np.pi,np.pi,30)
z = np.linspace(-0.1,0.1,30)


theta, z = np.meshgrid(theta, z)
x = r*np.cos(theta)
y = r*np.sin(theta)
cylinder_points = np.column_stack((x.ravel(),y.ravel(),z.ravel()))
rot_matrix = np.array([[0.5,0,0.866],[0.866,0,-0.5],[0,1,0]])
trans_matrix = np.array([[0.03,0.052,r]]).T
# trans_matrix = np.array([[0.03,0.035,r]]).T
transformed_pts = rot_matrix @ cylinder_points.T + trans_matrix
# index_closest = np.argmin(np.abs(transformed_pts[0,:]))
# transformed_cylinder_surf = transformed_pts.T[index_closest]
transformation_mat = np.concatenate((rot_matrix,trans_matrix),axis=1)
transformation_mat = np.concatenate((transformation_mat,np.array([[0,0,0,1]])))

thumb_nearby_pts, thumb_nearby_pts_angles = filter_points_near_cylinder(thumb_wksp.T,thumb_jt_angles,transformed_pts.T,rot_matrix[:,2],r,np.min(z),np.min(z)/1.5)
index_finger_nearby_pts, index_finger_nearby_pts_angles = filter_points_near_cylinder(index_finger_wksp.T,index_finger_jt_angles,transformed_pts.T,rot_matrix[:,2],r,np.min(z)/1.5,np.min(z)/1.85)
middle_finger_nearby_pts, middle_finger_nearby_pts_angles = filter_points_near_cylinder(middle_finger_wksp.T,middle_finger_jt_angles,transformed_pts.T,rot_matrix[:,2],r,np.min(z),np.max(z))
# ring_finger_nearby_pts, ring_finger_nearby_pts_angles = filter_points_near_cylinder(ring_finger_wksp.T,ring_finger_jt_angles,transformed_pts.T,rot_matrix[:,2],r,np.min(z)/8,np.max(z)/30)
ring_finger_proto2_nearby_pts, ring_finger_proto2_nearby_pts_angles = filter_points_near_cylinder(ring_finger_wksp_proto2.T,ring_finger_jt_angles_proto2,transformed_pts.T,rot_matrix[:,2],r,np.min(z)/8,np.max(z))
little_finger_nearby_pts, little_finger_nearby_pts_angles = filter_points_near_cylinder(little_finger_wksp.T,little_finger_jt_angles,transformed_pts.T,rot_matrix[:,2],r,np.min(z)/70,np.max(z)) # Use this for Max dof
# little_finger_nearby_pts, little_finger_nearby_pts_angles = filter_points_near_cylinder(little_finger_wksp.T,little_finger_jt_angles,transformed_pts.T,rot_matrix[:,2],r,np.min(z)/90,np.max(z)) # Use this for prototype 3
# little_finger_proto2_nearby_pts, little_finger_proto2_nearby_pts_angles = filter_points_near_cylinder(little_finger_wksp_proto2.T,little_finger_jt_angles_proto2,transformed_pts.T,rot_matrix[:,2],r,np.min(z)/50,np.max(z))

# print(thumb_nearby_pts.shape)
# print(index_finger_nearby_pts.shape)
# print(middle_finger_nearby_pts.shape)
# print(ring_finger_nearby_pts.shape)
# print(ring_finger_proto2_nearby_pts.shape)
# print(little_finger_nearby_pts.shape)
# print(little_finger_proto2_nearby_pts.shape)

# ring_nearby_pts_x_uvec = np.array(RF.fkine(ring_finger_nearby_pts_angles).A)[:,0:3,0].T
ring_proto2_nearby_pts_x_uvec = np.array(RF_proto2.fkine(ring_finger_proto2_nearby_pts_angles).A)[:,0:3,0].T
# little_nearby_pts_x_uvec = np.array(LF.fkine(little_finger_nearby_pts_angles).A)[:,0:3,0].T
# little_proto2_nearby_pts_x_uvec = np.array(LF_proto2.fkine(little_finger_proto2_nearby_pts_angles).A)[:,0:3,0].T

# little_F = little_nearby_pts_x_uvec
# little_F_proto2 = little_proto2_nearby_pts_x_uvec
# ring_F = ring_nearby_pts_x_uvec
ring_F_proto2 = ring_proto2_nearby_pts_x_uvec

origin = np.array([0.03,0.052,r])
weight_cylinder = 1
mu = 0.1

# F_mdofvsproto2_net = np.zeros(little_finger_nearby_pts.shape[0])
# F_mdofvsproto2_radial = np.zeros(little_finger_nearby_pts.shape[0])
# F_mdofvsproto2_axial = np.zeros(little_finger_nearby_pts.shape[0])

# F_mdofvsproto3_net = np.zeros(ring_finger_nearby_pts.shape[0])
# F_mdofvsproto3_radial = np.zeros(ring_finger_nearby_pts.shape[0])
# F_mdofvsproto3_axial = np.zeros(ring_finger_nearby_pts.shape[0])

# Fr = np.zeros(ring_finger_proto2_nearby_pts.shape[0])
# Fa = np.zeros(ring_finger_proto2_nearby_pts.shape[0])
# Fnet = np.zeros(ring_finger_proto2_nearby_pts.shape[0])

# for i in range(ring_finger_proto2_nearby_pts.shape[0]):
#     # ring_radial, ring_axial, _ = fc.decompose_force(ring_F[:,i],ring_finger_nearby_pts[i,:],origin,rot_matrix[:,2])
#     ring_proto2_radial, ring_proto2_axial, _ = fc.decompose_force(ring_F_proto2[:,i],ring_finger_proto2_nearby_pts[i,:],origin,rot_matrix[:,2])
#     # little_radial, little_axial, _ = fc.decompose_force(little_F[:,i],little_finger_nearby_pts[i,:],origin,rot_matrix[:,2])
#     # little_proto2_radial, little_proto2_axial, _ = fc.decompose_force(little_F_proto2[:,i],little_finger_proto2_nearby_pts[i,:],origin,rot_matrix[:,2])
#     # F_mdofvsproto2_radial[i] = mu*(little_radial - little_proto2_radial)
#     # F_mdofvsproto2_axial[i] = little_axial - little_proto2_axial
#     # F_mdofvsproto3_radial[i] = mu*(ring_radial - ring_proto2_radial)
#     # F_mdofvsproto3_axial[i] = ring_axial - ring_proto2_axial
#     # F_mdofvsproto3_net[i] = (mu*ring_radial + ring_axial - (mu*ring_proto2_radial + ring_proto2_axial))
#     # F_mdofvsproto2_net[i] = (mu*little_radial + little_axial - (mu*little_proto2_radial + little_proto2_axial))
#     Fa[i] = ring_proto2_axial
#     Fr[i] = ring_proto2_radial
#     Fnet[i] = ring_proto2_axial + mu*ring_proto2_radial - weight_cylinder

filePath = "c:/Users/Riswadkar/OneDrive - LTI/Documents/RWTH Robosys Master's Data/Semester 1/Hiwi codes and papers/Codes/Handkinematics/Shubhankar_codes/Test_codes/ring_finger_forces_Proto2.txt"
# littleF = True
# forces1 = np.array([Fa, Fr, Fnet])

# if littleF:
#     np.savetxt(filePath, forces1, delimiter=',')


content = np.loadtxt(filePath, delimiter=',')
z1 = np.argmax(content,axis=1)


th_ip = TH.fkine_all([0,-np.radians(47),0,0,0,0,0]).t
if_ip = IF.fkine_all([0,0,0,0,0,0,0]).t
mf_ip = MF.fkine_all([0,0,0,0,0]).t
# rf_ip = RF.fkine_all([0,0,0,0,0,0,0,0,0]).t
rf2_ip = RF_proto2.fkine_all([0,0,0,0,0,0,0,0]).t
lf_ip = LF.fkine_all([0,0,0,0,0,0,0,0,0]).t
# lf2_ip = LF_proto2.fkine_all([0,0,0,0,0,0,0,0]).t

fig2 = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(azim=-90, elev=90)
ax.scatter(ring_finger_proto2_nearby_pts[z1[0],0],ring_finger_proto2_nearby_pts[z1[0],1],ring_finger_proto2_nearby_pts[z1[0],2],marker='^',color='blue',label='Max axial force')
ax.scatter(ring_finger_proto2_nearby_pts[z1[1],0],ring_finger_proto2_nearby_pts[z1[1],1],ring_finger_proto2_nearby_pts[z1[1],2],marker='x',color='red',label='Max radial force')
ax.scatter(ring_finger_proto2_nearby_pts[z1[2],0],ring_finger_proto2_nearby_pts[z1[2],1],ring_finger_proto2_nearby_pts[z1[2],2],marker='o',color='green',label='Max net force')
# ax.scatter(transformed_pts[0,:],transformed_pts[1,:],transformed_pts[2,:],marker='o',color='cyan',label='Cylinder')
ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='g',marker='x',label="Thumb")
ax.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='orange',marker='x',label="LF")
# ax.plot(lf2_ip[:,0],lf2_ip[:,1],lf2_ip[:,2], c='orange',marker='x',label="LF")
ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
ax.plot(rf2_ip[:,0],rf2_ip[:,1],rf2_ip[:,2],c='red',marker='x',label="RF")
ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2], c='magenta',marker='x',label="MF")
plt.legend()
plt.tight_layout(pad=3.0)
plt.title("Max of forces in ring finger workspace close to cylinder surface for Prototype 3")
plt.savefig("Prototype 3 ring best grasping location",dpi=300)

# fig3 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(azim=-60, elev=30)
# ax.scatter(thumb_intersection_points_proto2[0,z2[0]],thumb_intersection_points_proto2[1,z2[0]],thumb_intersection_points_proto2[2,z2[0]],marker='^',color='blue',label='Max Euler Angle about X axis(Roll)')
# ax.scatter(thumb_intersection_points_proto2[0,z2[1]],thumb_intersection_points_proto2[1,z2[1]],thumb_intersection_points_proto2[2,z2[1]],marker='^',color='red',label='Max Euler Angle about Y axis(Pitch)')
# ax.scatter(thumb_intersection_points_proto2[0,z2[2]],thumb_intersection_points_proto2[1,z2[2]],thumb_intersection_points_proto2[2,z2[2]],marker='^',color='green',label='Max Euler Angle about Z axis(Yaw)')
# ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='g',marker='x',label="Thumb")
# ax.plot(lf2_ip[:,0],lf2_ip[:,1],lf2_ip[:,2], c='orange',marker='x',label="LF")
# ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
# ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2], c='magenta',marker='x',label="MF")
# plt.legend()
# plt.tight_layout(pad=3.0)
# plt.title("Max of euler angles in workspace intersection for Prototype2")
# plt.savefig("KOT optimal points 2",dpi=300)

# fig1 = plt.figure()
# plt.boxplot([content[2,:], F_mdofvsproto3_net], notch=False, patch_artist=True, showfliers=False,
#             boxprops=dict(facecolor='lightblue', color='blue'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='blue'),
#             medianprops=dict(color='red'))

# plt.xticks([1, 2], ['MAX DOF vs Prototype 2', 'Max DOF vs Prototype 3'])
# plt.title('Variation of the net force when grasping a cylinder')
# plt.ylabel('Net Force in axial direction (F)')
# # plt.savefig('Net force dibn.png', dpi=300)

# fig2 = plt.figure()
# plt.boxplot([content[0,:], F_mdofvsproto3_axial], notch=False, patch_artist=True, showfliers=False,
#             boxprops=dict(facecolor='lightblue', color='blue'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='blue'),
#             medianprops=dict(color='red'))

# plt.xticks([1, 2], ['MAX DOF vs Prototype 2', 'Max DOF vs Prototype 3'])
# plt.title('Variation of the axial force when grasping a cylinder')
# plt.ylabel('Axial Component of Force (F)')
# # plt.savefig('Axial force dibn.png', dpi=300)

# fig3 = plt.figure()
# plt.boxplot([content[1,:], F_mdofvsproto3_radial], notch=False, patch_artist=True, showfliers=False,
#             boxprops=dict(facecolor='lightblue', color='blue'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='blue'),
#             medianprops=dict(color='red'))

# plt.xticks([1, 2], ['MAX DOF vs Prototype 2', 'Max DOF vs Prototype 3'])
# plt.title('Variation of the radial force when grasping a cylinder')
# plt.ylabel('Radial Component of Force (F)')
# plt.savefig('Radial force dibn.png', dpi=300)

# fig1 = plt.figure()
# plt.boxplot([F_mdofvsproto2_net, content], notch=False, patch_artist=True, showmeans=True, meanline=True, showfliers=False,
#             boxprops=dict(facecolor='lightblue', color='blue'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='blue'),
#             medianprops=dict(color='lightblue'),
#             meanprops=dict(color='red'))
# plt.xticks([1,2], ['MAX DOF vs Prototype 2', 'MAX DOF vs Prototype 3'])
# plt.title('Variation of the net force when grasping a cylinder')
# plt.ylabel('Net Force in axial direction (F)')

# th_ip = TH.fkine_all([0,-np.radians(47),0,0,0,0,0]).t
# if_ip = IF.fkine_all([0,0,0,0,0,0,0]).t
# mf_ip = MF.fkine_all([0,0,0,0,0]).t
# rf_ip = RF.fkine_all([0,0,0,0,0,0,0,0,0]).t
# lf_ip = LF.fkine_all([0,0,0,0,0,0,0,0,0]).t
# lf2_ip = LF_proto2.fkine_all([0,0,0,0,0,0,0,0]).t

# fig1 = plt.figure()
# ax = plt.axes(projection='3d')
# # ax.view_init(azim=-121, elev=65)
# ax.view_init(azim=-90, elev=90)
# # ax.scatter(thumb_nearby_pts[:,0],thumb_nearby_pts[:,1],thumb_nearby_pts[:,2],marker='^',color='blue',label='Thumb')
# # ax.scatter(index_finger_nearby_pts[:,0],index_finger_nearby_pts[:,1],index_finger_nearby_pts[:,2],marker='^',color='green',label='Index')
# # ax.scatter(middle_finger_nearby_pts[:,0],middle_finger_nearby_pts[:,1],middle_finger_nearby_pts[:,2],marker='^',color='red',label='Middle')
# # ax.scatter(ring_finger_nearby_pts[:,0],ring_finger_nearby_pts[:,1],ring_finger_nearby_pts[:,2],marker='^',color='black',label='Ring')
# # ax.scatter(ring_finger_proto2_nearby_pts[:,0],ring_finger_proto2_nearby_pts[:,1],ring_finger_proto2_nearby_pts[:,2],marker='^',color='black',label='Ring')
# # ax.scatter(little_finger_nearby_pts[:,0],little_finger_nearby_pts[:,1],little_finger_nearby_pts[:,2],marker='^',color='magenta',label='Little')
# # ax.scatter(little_finger_proto2_nearby_pts[:,0],little_finger_proto2_nearby_pts[:,1],little_finger_proto2_nearby_pts[:,2],marker='^',color='magenta',label='Little')
# ax.scatter(transformed_pts[0,:],transformed_pts[1,:],transformed_pts[2,:],marker='o',color='cyan',label='Cylinder')
# ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='g',marker='x',label="Thumb")
# ax.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='orange',marker='x',label="LF")
# # ax.plot(lf2_ip[:,0],lf2_ip[:,1],lf2_ip[:,2], c='orange',marker='x',label="LF")
# ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
# ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2], c='magenta',marker='x',label="MF")
# plt.legend()
# # plt.title("Workspace points close to cylinder surface MAX DOF")
# plt.savefig("MAX DOF kinematic structure", dpi=300)
plt.show()