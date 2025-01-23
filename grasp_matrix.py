import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
import matplotlib.pyplot as plt
from spatialmath import SE3
from spatialmath.base import tr2rpy
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import block_diag
from scipy import optimize
from final_class_hand_kinematics import Handkinematics as hk 

def skew_symm_operator(r: np.ndarray):

    S = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])

    return S

def get_contact_frame(axis,center,appl_pt):

    t_hat = axis/np.linalg.norm(axis)

    r = appl_pt - center
    
    r_proj = np.dot(r,t_hat)*t_hat

    n = r - r_proj

    n_hat = -n/np.linalg.norm(n)

    s_hat = np.cross(n_hat,t_hat)

    return n_hat, s_hat, t_hat

def partial_grasp_matrix(orientation: np.ndarray, ci: np.ndarray, obj_centroid: np.ndarray):

    P = np.block([[np.eye(3), np.zeros((3,3))],[skew_symm_operator(r=ci-obj_centroid), np.eye(3)]])

    R = np.block([[orientation, np.zeros((3,3))],[np.zeros((3,3)), orientation]])

    Gt = R.T @ P.T

    return Gt

N_sample = 5000

TH_ROM = {'TH_CMC_a': -np.radians(47)-np.linspace(-np.radians(15),np.radians(60),N_sample),
      'TH_CMC_f': np.linspace(0,np.radians(60),N_sample),
      'TH_MCP_a': np.linspace(0,np.radians(30),N_sample),
      'TH_MCP_f': np.linspace(0,np.radians(55),N_sample),
      'TH_IP': np.linspace(0,np.radians(80),N_sample)}

IF_ROM = {'IF_MCP_a': np.linspace(-np.pi/6,np.pi/6,N_sample),
      'IF_MCP_f': np.linspace(0,(4*np.pi)/9,N_sample),
      'IF_PIP': np.linspace(0,(5*np.pi)/9,N_sample),
      'IF_DIP': np.linspace(0,np.pi/2,N_sample)}

MF_ROM = {'MF_MCP_a': np.linspace(-np.radians(22.5),np.radians(22.5),N_sample),
      'MF_MCP_f': np.linspace(0,np.radians(80),N_sample),
      'MF_PIP': np.linspace(0,np.radians(100),N_sample),
      'MF_DIP': np.linspace(0,np.pi/2,N_sample)}

RF_CMC_ROM = {'RF_CMC': np.linspace(0,np.radians(10),N_sample),
    'RF_MCP_a': np.linspace(-np.radians(22.5),np.radians(22.5),N_sample),
    'RF_MCP_f': np.linspace(0,np.radians(80),N_sample),
    'RF_PIP': np.linspace(0,np.radians(100),N_sample),
    'RF_DIP': np.linspace(0,np.pi/2,N_sample)}

RF_WCMC_ROM = {'RF_MCP_a': np.linspace(-np.radians(22.5),np.radians(22.5),N_sample),
    'RF_MCP_f': np.linspace(0,np.radians(80),N_sample),
    'RF_PIP': np.linspace(0,np.radians(100),N_sample),
    'RF_DIP': np.linspace(0,np.pi/2,N_sample)}

LF_CMC_ROM = {'LF_CMC': np.linspace(0,np.radians(20),N_sample),
    'LF_MCP_a': np.linspace(-np.radians(25),np.radians(25),N_sample),
    'LF_MCP_f': np.linspace(0,np.radians(80),N_sample),
    'LF_PIP': np.linspace(0,np.radians(100),N_sample),
    'LF_DIP': np.linspace(0,np.pi/2,N_sample)}

LF_WCMC_ROM = {'LF_MCP_a': np.linspace(-np.radians(25),np.radians(25),N_sample),
    'LF_MCP_f': np.linspace(0,np.radians(80),N_sample),
    'LF_PIP': np.linspace(0,np.radians(100),N_sample),
    'LF_DIP': np.linspace(0,np.pi/2,N_sample)}

th_initial_config = [-np.radians(47),0,0,0,0,0]
if_inital_config = [0,0,0,0,0,0,0]
mf_initial_config = [0,0,0,0,0]
rf_cmc_config = [0,0,0,0,0,0,0,0,0]
rf_wcmc_config = [0,0,0,0,0,0,0,0]
lf_cmc_config = [0,0,0,0,0,0,0,0,0]
lf_wcmc_config = [0,0,0,0,0,0,0,0]

th_params = [(0.028,0,0.02,np.radians(153.5),True,False),
            (0.0,np.pi/2,0.0,0.0,True,True),
            (0.048945,-np.radians(150),0.0,0.0,True,True),
            (0.0085,-np.pi/2,0.0,0.0,True,True),
            (0.03822,0.0,0.0,0.0,True,True),
            (0.03081,0.0,0.0,0.0,True,True)]

if_params = [(0.0,0.0,0.0,np.radians(106.5),True,False),
            (0.088,0.0,0.0,0.0,True,False), 
            (0.0085,0.0,0.0,0.0,True,True),
            (0.0,np.pi/2,0.0,-np.radians(10),True,False),
            (0.047775,0.0,0.0,0.0,True,True),
            (0.027885,0,0,0,True,True),
            (0.018915,0,0,0,True,True)]

mf_params = [(0.087,0.0,0.0,np.pi/2,True,False),
            (0.0085,np.pi/2,0.0,0.0,True,True),
            (0.05187,0.0,0.0,0.0,True,True),
            (0.03315,0.0,0.0,0.0,True,True),
            (0.018915,0.0,0.0,0.0,True,True)]

rf_mdof_params = [(0.0,0.0,0.0,np.radians(72.5),True,False),
                (0.031,np.pi/2,0.0,0.0,True,False),
                (0.0485,np.pi/2,0.0,0.0,True,True),
                (0.0085,-np.pi/2,0.0,0.0,True,True),
                (0.04758,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(7),True,False),
                (0.032175,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(7),True,False),
                (0.02865,0.0,0.0,0.0,True,True)]

rf_wcmc_params = [(0.0,0.0,0.0,np.radians(72.5),True,False),
                (0.0795,np.pi,0.0,0.0,True,False),
                (0.0085,-np.pi/2,0.0,0.0,True,True),
                (0.04758,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(7),True,False),
                (0.032175,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(7),True,False),
                (0.020865,0.0,0.0,0.0,True,True)]

lf_mdof_params = [(0.0,0.0,0.0,np.radians(63.5),True,False),
                (0.033,np.pi/2,0.0,0.0,True,False), 
                (0.043,np.pi/2,0.0,0.0,True,True),
                (0.0085,-np.pi/2,0.0,0.0,True,True),
                (0.03978,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(10),True,False),
                (0.022815,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(10),True,False),
                (0.018135,0.0,0.0,0.0,True,True)]

lf_wcmc_params = [(0.0,0.0,0.0,np.radians(63.5),True,False),
                (0.076,np.pi,0.0,0.0,True,False),
                (0.0085,-np.pi/2,0.0,0.0,True,True),
                (0.03978,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(10),True,False),
                (0.022815,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(10),True,False),
                (0.018135,0.0,0.0,0.0,True,True)]

proto1 = {'TH':(th_params, th_initial_config, TH_ROM),
    'IF': (if_params, if_inital_config, IF_ROM),
    'MF': (mf_params, mf_initial_config, MF_ROM),
    'RF': (rf_mdof_params, rf_cmc_config, RF_CMC_ROM),
    'LF': (lf_mdof_params, lf_cmc_config, LF_CMC_ROM)}

proto2 = {'TH':(th_params, th_initial_config, TH_ROM),
    'IF': (if_params, if_inital_config, IF_ROM),
    'MF': (mf_params, mf_initial_config, MF_ROM),
    'RF': (rf_wcmc_params, rf_wcmc_config, RF_WCMC_ROM),
    'LF': (lf_wcmc_params, lf_wcmc_config, LF_WCMC_ROM)}

hand1 = hk(hand_info=proto1)
hand1.compute_workspace()


# Cylinder surface parametrization
r = 0.03
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


h1_th_candidate_pts = hand1.filter_points_near_cylinder(finger='TH',cylinder_points=transformed_pts.T,cylinder_axis=rot_matrix[:,2],
                    cylinder_radius=r,cylinder_lowerlimit=np.min(z),cylinder_upper_limit=np.min(z)/1.4)
h1_if_candidate_pts = hand1.filter_points_near_cylinder(finger='IF',cylinder_points=transformed_pts.T,cylinder_axis=rot_matrix[:,2],
                    cylinder_radius=r,cylinder_lowerlimit=np.min(z)/1.5,cylinder_upper_limit=np.min(z)/1.85)
h1_mf_candidate_pts = hand1.filter_points_near_cylinder(finger='MF',cylinder_points=transformed_pts.T,cylinder_axis=rot_matrix[:,2],
                    cylinder_radius=r,cylinder_lowerlimit=np.min(z)/1.85,cylinder_upper_limit=np.max(z))
h1_rf_candidate_pts = hand1.filter_points_near_cylinder(finger='RF',cylinder_points=transformed_pts.T,cylinder_axis=rot_matrix[:,2],
                    cylinder_radius=r,cylinder_lowerlimit=np.min(z)/8,cylinder_upper_limit=np.max(z))
h1_lf_candidate_pts = hand1.filter_points_near_cylinder(finger='LF',cylinder_points=transformed_pts.T,cylinder_axis=rot_matrix[:,2],
                    cylinder_radius=r,cylinder_lowerlimit=-np.min(z)/70,cylinder_upper_limit=np.max(z))

h1_close_pts = (h1_th_candidate_pts[0],h1_if_candidate_pts[0],h1_mf_candidate_pts[0],h1_rf_candidate_pts[0],h1_lf_candidate_pts[0])

print(h1_lf_candidate_pts[0].shape, h1_if_candidate_pts[0].shape, h1_mf_candidate_pts[0].shape, h1_rf_candidate_pts[0].shape, h1_th_candidate_pts[0].shape)

hand1.plot_wksp_pts_close_to_cylinder(close_pts=h1_close_pts, cylinder_pts=transformed_pts,config='MAX DOF')
cylinder_center = np.mean(transformed_pts,axis=1)

n_l,s_l,t_l = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_lf_candidate_pts[0][1,:])
lf_frame = np.array([n_l,s_l,t_l]).reshape(9,1)
n_r,s_r,t_r = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_rf_candidate_pts[0][1,:])
rf_frame = np.array([n_r,s_r,t_r]).reshape(9,1)
n_m,s_m,t_m = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_mf_candidate_pts[0][1,:])
mf_frame = np.array([n_m,s_m,t_m]).reshape(9,1)
n_i,s_i,t_i = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_if_candidate_pts[0][1,:])
if_frame = np.array([n_i,s_i,t_i]).reshape(9,1)
n_th,s_th,t_th = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_th_candidate_pts[0][1,:])
th_frame = np.array([n_th,s_th,t_th]).reshape(9,1)

unit_vectors = np.squeeze(np.vstack((th_frame,if_frame,mf_frame,rf_frame,lf_frame)),axis=1)


G5 = partial_grasp_matrix(orientation=rot_matrix,ci=h1_lf_candidate_pts[0][1,:],obj_centroid=cylinder_center)
G4 = partial_grasp_matrix(orientation=rot_matrix,ci=h1_rf_candidate_pts[0][1,:],obj_centroid=cylinder_center)
G3 = partial_grasp_matrix(orientation=rot_matrix,ci=h1_mf_candidate_pts[0][1,:],obj_centroid=cylinder_center)
G2 = partial_grasp_matrix(orientation=rot_matrix,ci=h1_if_candidate_pts[0][1,:],obj_centroid=cylinder_center)
G1 = partial_grasp_matrix(orientation=rot_matrix,ci=h1_th_candidate_pts[0][1,:],obj_centroid=cylinder_center)

G_tilda_T = np.vstack((G1,G2,G3,G4,G5))
H = block_diag(np.eye(3),np.eye(3),np.eye(3),np.eye(3),np.eye(3))
H1 = block_diag(np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)))
H = np.hstack((H,H1))
G_T = H @ G_tilda_T
F = 10*rot_matrix[:,2].reshape(3,1)
w_ext = np.squeeze(np.vstack((F,np.zeros((3,1)))),axis=1)

def objective(f: np.ndarray):
    return np.dot(f.T,f)

def eq_fn(f: np.ndarray, cylinder_axis: np.ndarray, n_contacts: np.int32):

    f = f.flatten()
    G = G_T.T
    # F = 10*cylinder_axis.reshape(3,1)
    # w_ext = np.squeeze(np.vstack((F,np.zeros((3,1)))),axis=1)
    residual = (G@f - w_ext).flatten()
    cs1 = np.zeros(3*n_contacts+6)
    for i in range(n_contacts):
        # print(f"iter {i}")
        fi = f[3*i:3*i+3]
        # print(f"f shape {f.shape}")
        uvecs = unit_vectors[9*i:9*i+9]
        # print(f"u shape {uvecs.shape}")
        n = uvecs[0:3]
        # print(f"n shape {n.shape}")
        s = uvecs[3:6]
        t = uvecs[6:9]
        fn = np.dot(fi,n)
        ft = np.dot(fi,t)
        fs = np.dot(fi,s)
        cs1[3*i:3*i+3] = (fi - fn*n - ft*t - fs*s)
    
    cs1[3*n_contacts:] = residual
    assert cs1.shape[0] == 3 * n_contacts + 6, "Unexpected shape for constraints"
    assert residual.shape[0] == 6, "Unexpected shape for residual"
    print("Final constraints shape:", cs1.shape)
    print("Residual shape:", residual.shape)

    return cs1

def eq_jac(f: np.ndarray):
    G = G_T.T
    return G

eq_cons = {'type': 'eq', 'fun': lambda f: eq_fn(f, cylinder_axis=rot_matrix[:,2], n_contacts=5), 'jac': lambda f: eq_jac(f)}


lb = (-np.inf)*np.ones(15)
ub = 1e6*np.ones(15)
bounds = optimize.Bounds(lb,ub)
f0 = np.linalg.pinv(G_T.T) @ w_ext

print("Equality constraint at initial guess:", eq_fn(f0, cylinder_axis=rot_matrix[:, 2], n_contacts=5))
print("Bounds check (lower):", np.all(f0 >= lb))
print("Bounds check (upper):", np.all(f0 <= ub))
res = optimize.minimize(fun=objective,x0=f0,constraints=[eq_cons],method='SLSQP',bounds=bounds)

print(res)
# print(f"n_hat = {n}, s_hat = {s}, t_hat = {t}")

# arrow_length = 0.02  # Adjust this value to shorten the arrows

# fig1 = plt.figure()
# ax1 = plt.axes(projection="3d")
# ax1.scatter(transformed_pts[0,:],transformed_pts[1,:],transformed_pts[2,:])
# ax1.scatter(cylinder_center[0],cylinder_center[1],cylinder_center[2])
# ax1.quiver(transformed_pts[0,1], transformed_pts[1,1], transformed_pts[2,1], 
#         n[0], n[1], n[2], 
#         color='black', length=arrow_length, label='normal', linewidth=1)
# ax1.quiver(transformed_pts[0,1], transformed_pts[1,1], transformed_pts[2,1], 
#         s[0], s[1], s[2], 
#         color='magenta', length=arrow_length, label='sliding', linewidth=1)
# ax1.quiver(transformed_pts[0,1], transformed_pts[1,1], transformed_pts[2,1], 
#         t[0], t[1], t[2], 
#         color='gold', length=arrow_length, label='tangential', linewidth=1)
# ax1.quiver(cylinder_center[0], cylinder_center[1], cylinder_center[2], 
#         r1[0], r1[1], r1[2], 
#         color='g',label='pos vector', linewidth=1)
# ax1.quiver(cylinder_center[0], cylinder_center[1], cylinder_center[2], 
#         n1[0], n1[1], n1[2], 
#         color='b',label='unnormalized normal', linewidth=1)
# ax1.scatter(transformed_pts[0,1],transformed_pts[1,0],transformed_pts[2,1],c='g')

plt.legend()
plt.show()