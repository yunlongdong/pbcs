import pybullet as p
import pybullet_data
import numpy as np
import math
from matplotlib import pyplot as plt
from pbcs.trajectory_generation import ThirdPolyTraj


def getMotorJointStates(robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques

physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
p.setGravity(0, 0, -9.8)
planeId = p.loadURDF("plane.urdf")


startPos = [0, 0, 0.0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
ur5 = p.loadURDF('ur5/ur5.urdf', basePosition=startPos, baseOrientation=startOrientation, useFixedBase=True)
numj = p.getNumJoints(ur5)

control_joints = []
ee_link_index = 7
for i in range(numj):
    info = p.getJointInfo(ur5, i)
    joint_idx = info[0]
    joint_name = info[1]
    joint_type = info[2]
    link_name = info[12]
    if joint_type == 0:
        control_joints.append(joint_idx)

dt = 1/240
t = 0
total_T = 20

traj_T, traj_dt = np.linspace(0, total_T, 100, retstep=True)

circle_traj = ThirdPolyTraj(3)

xyz = np.array([0.5+0.2*np.sin(traj_T), 0+0.2*np.cos(traj_T), 0.3+0.1*np.sin(traj_T)])
dxyz = np.array([0.2*np.cos(traj_T), -0.2*np.sin(traj_T), 0.1*np.cos(traj_T)])

circle_traj.load_via_points(xyz, dxyz, traj_dt)

circle_traj.plot()

# p.setRealTimeSimulation(True)

ee_link_pos_serial = []
ee_link_vel_serial = []

desired_ee_link_pos_serial = []
desired_ee_link_vel_serial = []

# go to start
for i in range(100):
    q = p.calculateInverseKinematics(ur5, ee_link_index, targetPosition=[0.5, 0.2, 0.3], targetOrientation=p.getQuaternionFromEuler([0, 1.5707, 0]))
    p.setJointMotorControlArray(ur5, control_joints, p.POSITION_CONTROL, targetPositions=q)
    p.stepSimulation()

line_start_pos = p.getLinkState(ur5, ee_link_index)[0]

# trajctory velocity control
while True:
    desired_xyz, desired_dxyz = circle_traj.solution(t) 

    desired_ee_link_pos_serial.append(desired_xyz)
    desired_ee_link_vel_serial.append(desired_dxyz)

    desired_x, desired_y, desired_z = desired_xyz
    desired_dx, desired_dy, desired_dz = desired_dxyz
    
    actual_q = [ state[0] for state in p.getJointStates(ur5, control_joints)]
    
    linearJacobian, angularJacobian = p.calculateJacobian(ur5, linkIndex=ee_link_index, localPosition=[0, 0, 0], objPositions=actual_q, objVelocities=[0.]*len(actual_q), objAccelerations=[0.]*len(actual_q))

    jacobian = np.concatenate((np.array(linearJacobian), np.array(angularJacobian)), axis=0)

    desired_dq = np.linalg.pinv(linearJacobian).dot(desired_dxyz)

    desired_q = p.calculateInverseKinematics(ur5, ee_link_index, targetPosition=desired_xyz, targetOrientation=p.getQuaternionFromEuler([0, 1.5707, 0]))

    # feedforward and P feedback control
    joint_velocity = desired_dq + 10*(np.array(desired_q) - np.array(actual_q))

    p.setJointMotorControlArray(ur5, control_joints, p.VELOCITY_CONTROL, targetVelocities=joint_velocity)
    p.stepSimulation()

    ee_link_state = p.getLinkState(ur5, ee_link_index, computeLinkVelocity=1)
    ee_link_pos = ee_link_state[0]
    ee_link_vel = ee_link_state[6]

    ee_link_pos_serial.append(ee_link_pos)
    ee_link_vel_serial.append(ee_link_vel)


    # p.addUserDebugLine(line_start_pos, ee_link_pos, [1, 0, 0])
    line_start_pos = ee_link_pos
    t += dt
    if t > total_T:
        break

ee_link_pos = np.array(ee_link_pos_serial)
ee_link_vel = np.array(ee_link_vel_serial)

desired_ee_link_pos = np.array(desired_ee_link_pos_serial)
desired_ee_link_vel = np.array(desired_ee_link_vel_serial)

x = ee_link_pos[:, 0]
y = ee_link_pos[:, 1]
z = ee_link_pos[:, 2]

dx = ee_link_vel[:, 0]
dy = ee_link_vel[:, 1]
dz = ee_link_vel[:, 2]

desired_x = desired_ee_link_pos[:, 0]
desired_y = desired_ee_link_pos[:, 1]
desired_z = desired_ee_link_pos[:, 2]

desired_dx = desired_ee_link_vel[:, 0]
desired_dy = desired_ee_link_vel[:, 1]
desired_dz = desired_ee_link_vel[:, 2]

plt.subplot(211)
plt.plot(x, label='x')
plt.plot(desired_x, 'r-.', label='desired_x')
plt.legend()
plt.subplot(212)
plt.plot(dx, label='vx')
plt.plot(desired_dx, 'r-.', label='desired_vx')
plt.legend()
plt.show()


# fig = plt.figure(figsize = (10,10))
# ax = plt.axes(projection='3d')
# ax.plot3D(x,y,z,linewidth=2)
# ax.set_xlabel('x', labelpad=20)
# ax.set_ylabel('y', labelpad=20)
# ax.set_zlabel('z', labelpad=20)
# plt.show()