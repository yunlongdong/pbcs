import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
plt.style.use('seaborn-poster')


class ThirdPolyInterpolate:
    def __init__(self):
        self.poly_parameters = []

    def load_via_point(self, q, dq, dt=0.1):
        """
        q: postion vector
        dq: velocity vector, dq[0]=0, dq[-1]=0
        """
        self.q = np.asarray(q)
        self.dq = np.asarray(dq)
        self.dt = dt
        self.num_path = len(q)-1
    
    def interpolate(self):
        t = 0
        for i in range(self.num_path):
            start_q = self.q[i]
            start_dq = self.dq[i]

            end_q = self.q[i+1]
            end_dq = self.dq[i+1]
            
            tk = t
            tk1 = t + self.dt 

            time_matrix = np.array([[tk**3, tk1**3, 3*tk**2, 3*tk1**2], [tk**2, tk1**2, 2*tk, 2*tk1], [tk, tk1, 1, 1], [1, 1, 0, 0]])

            para_abcd = np.array([start_q, end_q, start_dq, end_dq]).dot(np.linalg.inv(time_matrix))
            self.poly_parameters.append(para_abcd)

            t += self.dt
        
    def solution(self, t):
        T = self.num_path*self.dt
        
        if t >= T:
            q = self.q[-1]
            dq = self.dq[-1] 
            return q, dq

        for i in range(self.num_path):
            a, b, c, d = self.poly_parameters[i]
            if t>=i*self.dt and t<(i+1)*self.dt:
                q = a*t**3 + b*t**2 + c*t + d
                dq = 3*a*t**2 + 2*b*t + c
                return q, dq
    
    def plot(self):
        T = self.num_path * self.dt

        t = np.linspace(0, T, 100)

        q = [self.solution(i)[0] for i in t]
        for i, iq in enumerate(self.q):
            plt.plot(i*self.dt, iq, 'ro')
        plt.plot(t, q)
        plt.show()    


class ThirdPolyTraj:
    def __init__(self, dof):
        self.dof = dof
        self.poly_interpolate_item = [ThirdPolyInterpolate() for i in range(self.dof)]

    def load_via_points(self, qs, dqs, dt):
        self.num_path = qs.shape[1]-1
        self.dt = dt
        assert qs.shape[0]==self.dof
        assert dqs.shape[0]==self.dof
        for i in range(self.dof):
            self.poly_interpolate_item[i].load_via_point(qs[i, :], dqs[i, :], dt)
            self.poly_interpolate_item[i].interpolate()

    def plot(self):
        for i in range(self.dof):
            plt.subplot(self.dof+1, 1, i+1)
            T = self.poly_interpolate_item[i].num_path * self.dt
            t = np.linspace(0, T, 100)
            q = [self.poly_interpolate_item[i].solution(j)[0] for j in t]
            plt.plot(t, q)
            for i, iq in enumerate(self.poly_interpolate_item[i].q):
                plt.plot(i*self.dt, iq, 'ro')
        plt.show()    

    def plotxyz(self):
        T = self.num_path*self.dt
        t = np.linspace(0, T, 500)
        x = [self.poly_interpolate_item[0].solution(j)[0] for j in t]
        y = [self.poly_interpolate_item[1].solution(j)[0] for j in t]
        z = [self.poly_interpolate_item[2].solution(j)[0] for j in t]
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection='3d')

        x_via_p = [self.poly_interpolate_item[0].q]
        y_via_p = [self.poly_interpolate_item[1].q]
        z_via_p = [self.poly_interpolate_item[2].q]

        ax.scatter(x_via_p, y_via_p, z_via_p, c = 'r', s = 40)

        ax.plot3D(x,y,z,linewidth=2)
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('z', labelpad=20)
        plt.show()

if __name__ == "__main__":
    traj_gen = ThirdPolyTraj(dof=6)

    t, dt = np.linspace(0, 6, 20, retstep=True)

    q = np.asarray([np.sin(t), np.cos(t), np.sin(t), 0*t, 0*t, 0*t])
    dq = np.asarray([np.cos(t), -np.sin(t), np.cos(t), 0*t, 0*t, 0*t])

    traj_gen.load_via_points(q, dq, dt=dt)

    # traj_gen.plotxyz()
    traj_gen.plot()

