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

table = p.loadURDF('model/table.urdf', basePosition=(0.7, 0, 0.2), baseOrientation=p.getQuaternionFromEuler([0, 0, 1.57]), useFixedBase=True)
ball = p.loadURDF('sphere_small.urdf', basePosition=(0.5, 0, 0.6), useFixedBase=True)

table_pos, table_ori = p.getBasePositionAndOrientation(table)
ball_pos, ball_ori = p.getBasePositionAndOrientation(ball)
ball_pos = list(ball_pos)
ball_pos.append(1)

table_trans = np.eye(4)
table_trans[:3, :3] = np.asarray(p.getMatrixFromQuaternion(table_ori)).reshape((3, 3))
table_trans[:3, 3] = table_pos

table_normal = table_trans[:3, 2]
nx, ny, nz = table_normal
print('table_normal:', table_normal)


# Pfaffian Constraints
A = np.array([[nx, ny, nz, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

print(A)

ball_pos_table = np.linalg.inv(table_trans).dot(np.array(ball_pos))
print('ball:', ball_pos, ball_pos_table[:3])

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

# go to start
for i in range(100):
    q = p.calculateInverseKinematics(ur5, ee_link_index, targetPosition=[0., 0., 0.5], targetOrientation=p.getQuaternionFromEuler([0, 0, 0]))
    p.setJointMotorControlArray(ur5, control_joints, p.POSITION_CONTROL, targetPositions=q)
    p.stepSimulation()

for i in control_joints:
    p.setJointMotorControl2(ur5, i, p.VELOCITY_CONTROL, force=0)
    # p.changeDynamics(ur5, i, jointDamping=0.2)
    p.stepSimulation()

p.enableJointForceTorqueSensor(ur5, 8, 1)

Fd = np.array([10, 0, 0, 0, 0, 0])
Xd = np.array([0.4, 0.2, 0.5, 0, 0, 0])

kp = np.array([200, 200, 200, 10, 2, 1])
kd = np.array([40, 40, 40, 0.5, 2, 0.05])

qe_interal = 0

qd = p.calculateInverseKinematics(ur5, ee_link_index, targetPosition=[0.4, 0, 0.3], targetOrientation=p.getQuaternionFromEuler([0, 1.5707, 0]))
qd = np.array(qd)

qe_serial = []

while True:
    # get ee force
    ee_force = p.getJointState(ur5, 8)[2]
    ee_force = np.array(ee_force)

    ee_link_state = p.getLinkState(ur5, ee_link_index, computeLinkVelocity=1)
    ee_link_pos = ee_link_state[0]
    ee_link_ori = ee_link_state[1]

    ee_trans = np.eye(4)
    ee_trans[:3, :3] = np.asarray(p.getMatrixFromQuaternion(ee_link_ori)).reshape((3, 3))
    ee_trans[:3, 3] = table_pos

    F_ee = list(ee_force)[:3] + [1]
    F_ee = np.array(F_ee)

    F_ee_base = np.linalg.inv(ee_trans).dot(np.array(F_ee))
    
    
    actual_q = [ state[0] for state in p.getJointStates(ur5, control_joints)]
    actual_dq = [ state[1] for state in p.getJointStates(ur5, control_joints)]

    # get jacobian
    linearJacobian, angularJacobian = p.calculateJacobian(ur5, linkIndex=ee_link_index, localPosition=[0, 0, 0], objPositions=actual_q, objVelocities=[0.]*len(actual_q), objAccelerations=[0.]*len(actual_q))
    J = np.concatenate((np.array(linearJacobian), np.array(angularJacobian)), axis=0)
    M = np.array(p.calculateMassMatrix(ur5, actual_q))

    CG = p.calculateInverseDynamics(ur5, actual_q, [0]*6, [0]*6)
    # $\Lambda$
    # L = np.linalg.pinv(J).T.dot(M).dot(np.linalg.pinv(J))
    # # projection matrix
    # L_inv = np.linalg.pinv(L)
    
    # P = np.eye(6) - A.T.dot(np.linalg.pinv(A.dot(L_inv).dot(A.T))).dot(A).dot(L_inv)

    # IP = np.eye(6) - P

    ee_link_state = p.getLinkState(ur5, ee_link_index, computeLinkVelocity=1)

    
    actual_q = np.array(actual_q)
    actual_dq = np.array(actual_dq)

    qe = qd - actual_q

    qe_serial.append(qe)

    qe_interal += qe

    # tau1 = M.dot(ddXd) + Kp*Xe 
    tau2 = kp*(qd - actual_q) + kd*(-actual_dq)

    tau = CG + tau2
    p.setJointMotorControlArray(ur5, control_joints, controlMode=p.TORQUE_CONTROL, forces=tau)
    p.stepSimulation()

    print(F_ee)
qe_serial = np.array(qe_serial)
print(qe_serial.shape)

for i in range(6):
    plt.subplot(6, 1, i+1)
    plt.plot(qe_serial[:, i])
    plt.grid()
plt.show()


