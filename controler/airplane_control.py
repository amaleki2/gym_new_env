import numpy as np
from gym.envs.registration import make

n_step_lqr, n_step_ilqr = 1500, 30
Q = np.eye(12, 12) *10
Qf = np.eye(12, 12) * 1000
R = np.eye(6, 6)
x_goal = [40., 0., 40., 0.,  0., 0., 0., .5, .5, 0., 0., 0.]
x_0    = [0.,  0., 0.,  2., -0., 0., .5, 0., 0., 0., 0., 0.]
x_med  = [10., 0., 10., 0., -0., 0., 0., 0., 0., 0., 0., 0.]
env = make("AirPlane-v0", dt=0.01, x_0=x_0, g=1.)
env.reset()

from util import ControllerAirPlane
control = ControllerAirPlane(env)
ilqr_actions = control.run_ilqr(Q, R, Qf, x_med, n_step_ilqr)
control.run_lqr(Q, R, x_goal, n_step_lqr, ilqr_actions[-1])

env.render(skip=5)
#env.animate(file_name="ap.gif", x_goal=x_goal)
print([h['x'] for h in env.history])
