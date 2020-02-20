import gym
import pickle
import numpy as np
from numpy import sin, cos, pi
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# This is a 3-link pendulum.

class Pendulum2Env(gym.Env):
    """
    """
    """
    this is a 2-link pendulum with identical links. The joins have no mass. There is viscous friction assumed. The inputs are
        -m: mass of each link
        -L: length of each link
        -J: moment of inertia of each link
        -c: coefficient of viscous friction
        -th_0: initial angles
        -u_0: initial angular velocities.
        -u1, u2: torque at joints
        
        For the governing equation, it is assumed th1 and th2 are angles made by the first and second links and 
        vertical axis. 

    The equations are extracted from  http://www.cs.cmu.edu/~mpalatuc/kdc/hw2/ 
    """

    """
    IMPORTATNT NOTE:
    ** th1 and th2 are measured from upright vertical axis; i.e. when the pendulum is upright. th1 and th2 are both 0.
    ** th2 is NOT relative to th1. 
    ** positive y axis is looking downward. hence g= POSITIVE 9.8
    """
    # metadata = {
    #     'render.modes' : ['human', 'rgb_array'],
    #     'video.frames_per_second' : 30
    # }

    def __init__(self, m=0.5, L=0.5, J=0.334167, c=0.1, th_0=[0., 0.], u_0=[0., 0.],
                       max_torque=1000, g=9.8, dt=0.0001, N=1, init_guess = [0., 0.]):
        self.m,    self.L,   self.J, self.c = m, L, J, c
        self.g,    self.dt,  self.N, self.max_torque = g, dt, N, max_torque
        self.th_0, self.u_0, self.init_guess = th_0, u_0, init_guess
        self.th,   self.u,   self.current_guess, self.history = [None]*4 # these will be updated in the reset.
        self.integration_method = "1st"
        self.reset()

        self.a1 = self.m*self.L**2
        self.a2 = 0.25*self.m*self.L**2 + 2*self.J + 2*self.m*self.L**2
        self.a3 = 0.25*self.m*self.L**2 + self.J
        self.a4 = self.m*self.g*self.L

    def reset(self):
        self.th = np.array(self.th_0)
        self.u  = np.array(self.u_0)
        self.current_guess = self.init_guess
        self.history = [{"th": self.th.copy(), "reward": 0, "u": self.u.copy(), "actions": (0, 0)}]

    def setup_equation(self, u_prime, th, u, actions):
        T1, T2 = actions
        th1, th2 = th
        u1, u2 = u
        u1_prime, u2_prime = u_prime

        # eq1 = (self.m*self.L**2/4 + self.J + 2*self.m*self.L**2 + self.m*self.L**2*cos(th2) + self.J)*u1p \
        #     + (self.m*self.L**2/4 + self.J + self.m*self.L**2*cos(th2)/2)*u2p \
        #     - (self.m*self.L**2*sin(th2)*u1*u2) - (self.m*self.L**2*sin(th2)*u2**2/2) \
        #     + (self.m*self.L*self.g*sin(th1)/2) + (self.m*self.L*self.g*sin(th1)) + (self.m*self.L*self.g*sin(th1+th2)/2) \
        #     - T1 + self.c*u1
        # eq2 = (self.m*self.L**2/4 + self.m*self.L**2*cos(th2)/2 + self.J)*u1p \
        #     + (self.m*self.L**2/4 + self.J)*u2p \
        #     + (self.m*self.L**2*sin(th2)*u1**2/2) \
        #     + (self.m*self.L*self.g*sin(th1+th2)/2) \
        #     - T2 + self.c*u2

        # eq1 = (self.a1*cos(th2))*u1p + (self.a3 + 0.5*self.a1*cos(th2))*u2p - self.a1*sin(th2)*u1*u2 + \
        #       self.a4*sin(th1) + 0.5*self.a4*sin(th1) + 0.5*self.a4*sin(th1+th2) - 0.5*self.a1*sin(th2)*u2*u2 - T1 + self.c*u1
        # eq2 = (self.a3 + 0.5*self.a1*cos(th2))*u1p + self.a3*u2p + 0.5*self.a1*sin(th2)*u1*u1 + 0.5*self.a4*sin(th1+th2) - T2 + self.c*u2

        eq1 = 2*u1_prime + u2_prime*cos(th2 - th1) - u2**2*sin(th2 - th1) - self.g/self.L*sin(th1)*2 - T1/(self.m*self.L**2) + self.c*u1/(self.m*self.L**2)
        eq2 =   u2_prime + u1_prime*cos(th2 - th1) + u1**2*sin(th2 - th1) - self.g/self.L*sin(th2)   - T2/(self.m*self.L**2) + self.c*u2/(self.m*self.L**2)

        return eq1, eq2

    def step(self, actions):
        actions = np.clip(actions, -self.max_torque, self.max_torque).tolist()
        costs = 0
        for _ in range(self.N):
            if self.integration_method == "1st":
                u_prime = fsolve(self.setup_equation, self.current_guess, args=(self.th, self.u, actions))
            else:
                u_prime_half = fsolve(self.setup_equation, self.current_guess, args=(self.th, self.u, actions))
                u_half   = self.u  + 1/2*self.dt * u_prime_half
                th_half  = self.th + 1/2*self.dt * u_half  - 1/8*u_prime_half*self.dt**2
                u_prime = fsolve(self.setup_equation, self.current_guess, args=(th_half, u_half, actions))
            self.u  += self.dt*u_prime
            self.th += self.dt*self.u - 1/2*u_prime*self.dt**2
            # costs += 1.000*np.sum(np.square([min(angle_normalize(th-pi), angle_normalize(th+pi)) for th in self.th])) #+ \
            #         # 0.100*np.sum(np.square(self.u)) + \
            #         # 0.001*np.sum(np.square(u_p))

        self.history.append({"th": self.th.copy(), "reward": -costs, "u": self.u.copy(), "actions": actions})
        return self.th, -costs, False, {}

    def save_history(self, name="history.pkl"):
        with open(name, 'wb') as fid:
            pickle.dump(self.history, fid, protocol=pickle.HIGHEST_PROTOCOL)

    def render(self, mode="human", read_file=False, close=False, skip=1):
        if read_file:
            with open("history.pkl", "rb") as fid:
                self.history = pickle.load(fid)

        plt.figure()
        for i in range(0, len(self.history), skip):
            plt.cla()

            u       = self.history[i]["u"]
            th      = self.history[i]["th"]
            reward  = self.history[i]["reward"]
            actions = self.history[i]["actions"]
            x0, y0  = 0, 0
            x1, y1  = x0 - self.L*sin(th[0]), y0 - self.L*cos(th[0])
            # x2, y2  = x1 + self.L*sin(th[1]+th[0]), y1 + self.L*cos(th[1]+th[0])
            x2, y2  = x1 - self.L*sin(th[1]), y1 - self.L*cos(th[1])
            plt.plot([0,  x1], [0,  y1], color='black')
            plt.plot([x1, x2], [y1, y2], color='black', markersize=10, markerfacecolor='red', marker = "o")




            limits = self.L*2.2
            xarrow, yarrow = -0.9 * limits, 0
            plt.arrow(xarrow, yarrow, limits*0, limits*0.2, width=0.02, head_width=0.06, color="black")
            plt.xlim([-limits, limits])
            plt.ylim([limits, -limits])
            plt.text(xarrow*0.95, yarrow + limits*0.2, "g")

            xtext, ytest = 0.1*limits, 0.8*limits
            plt.text(xtext, ytest, "th1:%0.1f, th2:%0.1f, u1:%0.1f, u2:%0.1f" %(th[0], th[1], u[0], u[1]))
            xtext, ytest = 0.1*limits, 0.65*limits
            plt.text(xtext, ytest, "actions=(%0.2f, %0.2f), reward:%0.2f" %(actions[0], actions[1], reward))
            plt.draw()
            plt.pause(0.01)
        if close:
            plt.close()

    def animate(self, skip=1):
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)

        # Plot a scatter that persists (isn't redrawn) and the initial line.
        th = self.history[0]["th"]
        x0, y0 = 0, 0
        x1, y1 = x0 + self.L * sin(th[0]), y0 + self.L * cos(th[0])
        x2, y2 = x1 + self.L * sin(th[1]), y1 + self.L * cos(th[1])

        line1, = ax.plot([0, x1], [0, y1], color='black')
        line2, = ax.plot([x1, x2], [y1, y2], color='black', markersize=10, markerfacecolor='red', marker="o")
        limits = self.L*2.2
        plt.xlim([-limits, limits])
        plt.ylim([-limits, limits])
        def update(i):
            label = 'timestep {0}'.format(i)
            print(label)
            u = self.history[i]["u"]
            th = self.history[i]["th"]
            reward = self.history[i]["reward"]
            actions = self.history[i]["actions"]
            x0, y0 = 0, 0
            x1, y1 = x0 + self.L * sin(th[0]), y0 + self.L * cos(th[0])
            x2, y2 = x1 + self.L * sin(th[1]), y1 + self.L * cos(th[1])
            # Update the line and the axes (with a new xlabel). Return a tuple of
            # "artists" that have to be redrawn for this frame.
            line1.set_xdata([0, x1])
            line1.set_ydata([0, y1])
            line2.set_xdata([x1, x2])
            line2.set_ydata([y1, y2])
            ax.set_xlabel(label)
            return line1, line2, ax

        anim = FuncAnimation(fig, update, frames=np.arange(0, len(self.history), skip), interval=50)
        anim.save('line.gif', dpi=80, writer='imagemagick')
        plt.show()


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)