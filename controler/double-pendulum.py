import numpy as np
from util import ControllerDoublePendulum
from gym.envs.registration import make


env = make("Pendulum2-v0", x_0=[np.pi*0.9, -np.pi*0.8, 0., -1.], dt=0.01)
env.reset()
control = ControllerDoublePendulum(env)
n_step_lqr, n_step_ilqr = 250, 150
Q = np.eye(4, 4); Q[1, 1] = 0; Q[2, 2] = 0
Qf = np.eye(4, 4)*1000
R = np.eye(2, 2)
x_goal = [0., 0., 0., 0.]

ilqr_actions = control.run_ilqr(Q, R, Qf, x_goal, n_step_ilqr)
lqr_actions  = control.run_lqr(Q, R, x_goal, n_step_lqr, ilqr_actions[-1])
print(env.x)
env.render()

