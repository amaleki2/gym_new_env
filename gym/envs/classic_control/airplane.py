import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
ORIGIN_PATH = "/home/amaleki/Dropbox/stanford/Python/gym-new-env/gym/envs/classic_control"

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],facecolor=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


class AirplaneEnv(gym.Env):
    """
        this is a dynamical system for an airplane
    """
    def __init__(self,
                 m=1., Ix=1., Iy=1., Iz=1., Ixz=0.5, Ixx =0.5, g=9.8,  # physical paramters
                 x_0=0.,  # initializations
                 dt=0.01, # computational parameters.
                 max_action = 100
                 ):
        self.n_state, self.n_action = 12, 6
        self.m, self.Ix, self.Iy, self.Iz, self. Ixz = m, Ix, Iy, Iz, Ixz
        self.g, self.dt = g, dt
        self.max_action = max_action
        if hasattr(x_0, "__len__"):
            self.x_0 = x_0
        else:
            self.x_0 = np.ones(self.n_state) * x_0
        print(os.getcwd())

        self.airplane_xy_0 = np.loadtxt(os.path.join(ORIGIN_PATH, "plane_xy.txt")).T
        self.airplane_xy_0[0, :] /= 42.
        self.airplane_xy_0[1, :] *= -1.
        self.airplane_xz_0 = np.loadtxt(os.path.join(ORIGIN_PATH, "plane_xz.txt")).T
        self.airplane_xz_0[1, :] *= -1.
        self.airplane_yz_0 = np.loadtxt(os.path.join(ORIGIN_PATH, "plane_yz.txt")).T
        self.airplane_yz_0[1, :] *= -1.

        self.x, self.history = [None] * 2  # these will be updated in the reset.

        self.reset()

    def reset(self):
        self.x = np.array(self.x_0)
        self.history = [{"x": self.x.copy(), "a": [0.] * (self.n_action), "r": 0.}]

    @staticmethod
    def rot_matrices(phi, theta, psi):
        T_psi = np.array([
            [np.cos(psi), -np.sin(psi), 0.],
            [np.sin(psi), np.cos(psi), 0.],
            [0., 0., 1.]
        ])
        T_theta = np.array([
            [np.cos(theta), 0., np.sin(theta)],
            [0., 1., 0.],
            [-np.sin(theta), 0., np.cos(theta)]
        ])
        T_phi = np.array([
            [1., 0., 0.],
            [0., np.cos(phi), -np.sin(phi)],
            [0., np.sin(phi), np.cos(phi)]
        ])

        mat_1 = np.matmul(T_psi, T_theta, T_phi)

        mat_2 = np.array([
            [np.cos(theta), np.sin(theta) * np.sin(phi),  np.sin(theta) * np.cos(phi)],
            [0.,            np.cos(theta) * np.cos(phi), -np.cos(theta) * np.sin(phi)],
            [0.,                            np.sin(phi),                  np.cos(phi)]
        ])
        mat_2 = 1 / np.cos(theta) * mat_2

        return mat_1, mat_2

    def dynamics(self, states, t, actions):

        # unwraping states and actions
        x, y, z, u, v, w, phi, theta, psi, p, q, r = states
        X, Y, Z, L, M, N = actions

        # computing rotation matrices
        mat_1, mat_2 = self.rot_matrices(phi, theta, psi)

        # temporarily variables
        a1 = np.array([u, v, w]).T
        a2 = mat_1.dot(a1)
        a3 = np.array([p, q, r]).T
        a4 = mat_2.dot(a3)
        a5 = np.array([
            [self.Ix, self.Ixz],
            [self.Ixz, self.Iz]
        ])
        a6 = np.array([
            [L - (self.Iz - self.Iy) * q * r - self.Ixz * q * p],
            [N - (self.Iy - self.Ix) * q * p + self.Ixz * q * r]
        ])
        a7 = np.linalg.inv(a5).dot(a6)

        # computing the derivatives
        dx = a2[0]
        dy = a2[1]
        dz = a2[2]

        du = -self.g * np.sin(theta) + X / self.m - q * w + r * v
        dv =  self.g * np.cos(theta) * np.sin(phi) + Y / self.m - r * u + p * w
        dw =  self.g * np.cos(theta) * np.cos(phi) + Z / self.m - p * v + q * u

        dphi = a4[0]
        dthe = a4[1]
        dpsi = a4[2]

        dp = a7[0][0]
        dq = 1./self.Iy *(M - self.Ixz * (r ** 2 - p ** 2) - (self.Ix - self.Iz) * p * r)
        dr = a7[1][0]

        dXdt = np.array([dx, dy, dz, du, dv, dw, dphi, dthe, dpsi, dp, dq, dr])
        return dXdt

    def step(self, actions):
        self.x = odeint(self.dynamics, self.x, [0, self.dt], args=(actions, ))[-1, :]
        self.history.append({"x": self.x.copy(), "r": 0, "a": actions})
        return self.x, 0, False, {}

    def render(self, mode="human", skip=1):
        fig, ax = plt.subplots(figsize=(10, 8))
        rect_xy = [0.1, 0.1, 0.2, 0.195]
        ax_xy = add_subplot_axes(ax, rect_xy)
        rect_yz = [0.1, 0.5, 0.2, 0.195]
        ax_yz = add_subplot_axes(ax, rect_yz)
        for i in range(0, len(self.history), skip):
            x, y, z, u, v, w, phi, theta, psi, p, q, r = self.history[i]["x"]

            rot_mat_xy = np.array([[np.cos(-psi), -np.sin(-psi)],
                                   [np.sin(-psi),  np.cos(-psi)]
                                  ])
            rot_mat_xz = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta),  np.cos(theta)]
                                   ])
            rot_mat_yz = np.array([[np.cos(phi), -np.sin(phi)],
                                   [np.sin(phi),  np.cos(phi)]
                                   ])

            airplane_xy = rot_mat_xy.dot(self.airplane_xy_0)
            airplane_yz = rot_mat_yz.dot(self.airplane_yz_0)

            airplane_xz = rot_mat_xz.dot(self.airplane_xz_0)
            airplane_xz[0, :] += x
            airplane_xz[1, :] += z

            ax.cla()
            ax.fill(airplane_xz[0, :], airplane_xz[1, :])
            ax.set_aspect('equal', 'box')
            xlim, ylim = 50, 50
            ax.set_xlim([-xlim, xlim])
            ax.set_ylim([ylim, -ylim])
            #ax.draw()

            ax_xy.cla()
            ax_xy.fill(airplane_xy[0, :], airplane_xy[1, :])
            xlim, ylim = 50, 20
            ax_xy.set_xlim([0, xlim])
            ax_xy.set_ylim([ylim, -ylim])
            ax_xy.set_aspect('equal', 'box')
            ax_xy.set_axis_off()

            ax_yz.cla()
            ax_yz.fill(airplane_yz[0, :], airplane_yz[1, :])
            xmin = airplane_yz[0, :].min() - 10
            ymin = airplane_yz[1, :].min() - 50
            ax_yz.set_xlim([xmin, xmin+150])
            ax_yz.set_ylim([ymin+150, ymin])
            ax_yz.set_aspect('equal', 'box')
            ax_yz.set_axis_off()
            plt.draw()
            plt.pause(0.01)

    def animate(self, file_name="airplane.gif", skip=1, x_goal=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.set_tight_layout(True)
        rect_xy = [0.1, 0.1, 0.2, 0.195]
        ax_xy = add_subplot_axes(ax, rect_xy)
        rect_yz = [0.1, 0.5, 0.2, 0.195]
        ax_yz = add_subplot_axes(ax, rect_yz)

        x, y, z, u, v, w, phi, theta, psi, p, q, r = self.history[0]["x"]
        rot_mat_xy = np.array([[np.cos(-psi), -np.sin(-psi)],
                               [np.sin(-psi), np.cos(-psi)]
                               ])
        rot_mat_xz = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]
                               ])
        rot_mat_yz = np.array([[np.cos(phi), -np.sin(phi)],
                               [np.sin(phi), np.cos(phi)]
                               ])

        airplane_xy = rot_mat_xy.dot(self.airplane_xy_0)
        airplane_yz = rot_mat_yz.dot(self.airplane_yz_0)

        airplane_xz = rot_mat_xz.dot(self.airplane_xz_0)
        airplane_xz[0, :] += x
        airplane_xz[1, :] += z

        fill1, = ax.fill(airplane_xz[0, :], airplane_xz[1, :])
        ax.set_aspect('equal', 'box')
        xlim, ylim = 50, 50
        ax.set_xlim([-xlim, xlim])
        ax.set_ylim([ylim, -ylim])

        fill2, = ax_xy.fill(airplane_xy[0, :], airplane_xy[1, :])
        xlim, ylim = 50, 20
        ax_xy.set_xlim([0, xlim])
        ax_xy.set_ylim([ylim, -ylim])
        ax_xy.set_aspect('equal', 'box')
        ax_xy.set_axis_off()

        fill3, = ax_yz.fill(airplane_yz[0, :], airplane_yz[1, :])
        xmin = airplane_yz[0, :].min() - 10
        ymin = airplane_yz[1, :].min() - 50
        ax_yz.set_xlim([xmin, xmin + 150])
        ax_yz.set_ylim([ymin + 150, ymin])
        ax_yz.set_aspect('equal', 'box')
        ax_yz.set_axis_off()

        if x_goal is not None:
            ax.plot(x_goal[0], x_goal[2], color='red', marker='*', markersize=10)

        def update(i):
            label = 'timestep {0}'.format(i)
            print(label)
            x, y, z, u, v, w, phi, theta, psi, p, q, r = self.history[i]["x"]

            rot_mat_xy = np.array([[np.cos(-psi), -np.sin(-psi)],
                                   [np.sin(-psi), np.cos(-psi)]
                                   ])
            rot_mat_xz = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]
                                   ])
            rot_mat_yz = np.array([[np.cos(phi), -np.sin(phi)],
                                   [np.sin(phi), np.cos(phi)]
                                   ])

            airplane_xy = rot_mat_xy.dot(self.airplane_xy_0)
            airplane_yz = rot_mat_yz.dot(self.airplane_yz_0)

            airplane_xz = rot_mat_xz.dot(self.airplane_xz_0)
            airplane_xz[0, :] += x
            airplane_xz[1, :] += z

            fill1.set_xy(np.array([airplane_xz[0, :], airplane_xz[1, :]]).T)
            ax.set_xlabel(label)

            fill2.set_xy(np.array([airplane_xy[0, :], airplane_xy[1, :]]).T)

            fill3.set_xy(np.array([airplane_yz[0, :], airplane_yz[1, :]]).T)
            return fill1, fill2, fill3, ax

        anim = FuncAnimation(fig, update, frames=np.arange(0, len(self.history), skip), interval=20)
        if file_name is not None:
            anim.save(file_name, dpi=30, writer='imagemagick')
        plt.show()


from gym.envs.registration import make
if __name__ == '__main__':
    env = make("AirPlane-v0", g=1.)
    env.reset()
    actions = [0.5, 0., -1.0, 0., 0., 0.]

    for i in range(2000):
        actions[3] = np.random.randn()*.2
        env.x, _, _, _ = env.step(actions)
    env.render(skip=5)


np.random.rand(5)

