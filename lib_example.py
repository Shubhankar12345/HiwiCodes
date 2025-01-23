import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
import matplotlib.pyplot as plt
from spatialmath import SE3
from spatialmath.base import tr2rpy
from sklearn.metrics.pairwise import euclidean_distances
from final_class_hand_kinematics import Handkinematics as hk 

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
hand2 = hk(hand_info=proto2)
hand1.compute_workspace()
# hand2.compute_workspace()
# hand1.plot_finger_workspace('TH')
# hand2.plot_finger_workspace('TH')
th_mdof, th_mdof_config, opp_mdof, opp_mdof_config = hand1.compute_workspace_intersection_points(thumb='TH',opposing_finger='LF')
# th_pr2, th_pr2_config, opp_pr2, opp_pr2_config = hand2.compute_workspace_intersection_points(thumb='TH',opposing_finger='LF')
# mdof_orientn_sim, mdof_orientn_euler = hand1.KOTcomputation(th_angles=th_mdof_config,oppfinger_angles=opp_mdof_config,oppfinger='LF')
# proto2_orientn_sim, proto2_orientn_euler = hand2.KOTcomputation(th_angles=th_pr2_config,oppfinger_angles=opp_pr2_config,oppfinger='LF')
# hand1.plot_wksp_intersection(th_wksp=th_mdof,oppfinger_wksp=opp_mdof,config='MAX DOF')

index1 = np.random.choice(th_mdof_config.shape[0])
th_anim_config = th_mdof_config[index1,:]
opp_anim_config = opp_mdof_config[index1,:]
hand1.animate_hand(th_jt_config=th_anim_config,opp_jt_config=opp_anim_config,oppfinger='LF')

# Adjust these values to fine-tune the reference
# xmin1 = 0.2
# xmax1 = 0.3


# xmin2 = 0.7
# xmax2 = 0.8

# meanprops = {
#     'marker': '^',        
#     'markerfacecolor': 'red',
#     'markeredgecolor': 'black', 
#     'markersize': 8
# }

# fig2 = plt.figure()
# plt.boxplot([mdof_orientn_sim, proto2_orientn_sim], notch=False, patch_artist=True, showmeans=True, showfliers=False,
#             boxprops=dict(facecolor='lightblue', color='blue'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='blue'),
#             medianprops=dict(color='red'),
#             meanprops=meanprops)
# plt.axhline(y=2.82842712474619, color='k', linestyle='--', xmin=xmin1, xmax=xmax1, label='Desired Orientation Mismatch')
# plt.axhline(y=2.82842712474619, color='k', linestyle='--', xmin=xmin2, xmax=xmax2)
# plt.xticks([1, 2], ['MAX DOF', 'Prototype 2'])
# plt.title('Variation of the orientation between thumb and opposing fingers fingertip')
# plt.ylabel('Orientation mismatch')
# plt.legend()
# plt.savefig('Orientation Mismatch KOT.png', dpi=300)

plt.show()