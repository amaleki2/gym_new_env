import gym
import tqdm
import scipy
import numpy as np
import sympy as sym
from sympy import sin, cos
from scipy.integrate import odeint


class ControllerPendulum2():
    def __init__(self, env, method=1):
        self.env = env
        self.A, self.B, self.Q, self.R, self.vars = [None]*5 # These attributes will be initialized below.
        self.compute_matrices(method)

    def compute_matrices(self, method):
        th1, th2 = sym.symbols('th1, th2')
        u1, u2 = sym.symbols('u1, u2')
        T1, T2 = sym.symbols('T1, T2')

        M = sym.Matrix([[2, cos(th2 - th1)],
                       [cos(th2 - th1), 1]])
        C = sym.Matrix([[ sin(th2 - th1)*u2**2],
                        [-sin(th2 - th1)*u1**2]])
        G = self.env.g/self.env.L*sym.Matrix([[sin(th1)*2],
                                              [sin(th2)]])
        T = 1./(self.env.m*self.env.L**2)*sym.Matrix([[T1],
                                                      [T2]])
        F = -1.*self.env.c/(self.env.m*self.env.L**2)*sym.Matrix([[u1],
                                                                  [u2]])
        # m, L, J, c, g = self.env.m, self.env.L, self.env.J, self.env.c, self.env.g
        # M = sym.Matrix([[ m*L**2/4 + J + 2*m*L**2 + m*L**2*cos(x2) + J, m*L**2/4 + J + m*L**2*cos(x2)/2],
        #                [ m*L**2/4 + m*L**2*cos(x2)/2 + J,            m*L**2/4 + J]])
        # C = sym.Matrix([[-m*L**2*sin(x2)*u1*u2 - m*L**2*sin(x2)*u2**2/2],
        #                [ m*L**2*sin(x2)*u1**2/2]])
        # G = sym.Matrix([[ m*L*g*sin(x1)/2 + m*L*g*sin(x1) + m*L*g*sin(x1+x2)/2],
        #                [ m*L*g*sin(x1+x2)/2]])
        # F = sym.Matrix([[ c*u1],
        #                [ c*u2]])
        # T = sym.Matrix([[ T1],
        #                [ T2]])


        if method == 1:
            aMat = M.inv()*(T + F + C + G)
            q0 = aMat.diff(th1)
            q1 = aMat.diff(th2)
            q2 = aMat.diff(u1)
            q3 = aMat.diff(u2)
            q4 = aMat.diff(T1)
            q5 = aMat.diff(T2)

            self.A = sym.Matrix([[0,     0,     1,     0],
                                 [0,     0,     0,     1],
                                 [q0[0], q1[0], q2[0], q3[0]],
                                 [q0[1], q1[1], q2[1], q3[1]]])

            self.B = sym.Matrix([[0,     0],
                                 [0,     0],
                                 [q4[0], q5[0]],
                                 [q4[1], q5[1]]])
        # else:
        #     M_inv = M.inv()
        #     aMat = sym.sympify(M_inv*(F + C + G)*(-1))
        #     q0 = aMat.diff(x1)
        #     q1 = aMat.diff(x2)
        #     q2 = aMat.diff(u1)
        #     q3 = aMat.diff(u2)
        #
        #     self.A = sym.Matrix([[0, 0, 1, 0],
        #                          [0, 0, 0, 1],
        #                          [q0[0], q1[0], q2[0], q3[0]],
        #                          [q0[1], q1[1], q2[1], q3[1]]])
        #
        #     self.B = sym.zeros(4,2)
        #     self.B[2, 0] = M_inv[0, 0]
        #     self.B[2, 1] = M_inv[0, 1]
        #     self.B[3, 0] = M_inv[1, 0]
        #     self.B[3, 1] = M_inv[1, 1]

        self.Q = sym.eye(4, 4)
        self.Q[1, 1] = 0
        self.Q[2, 2] = 0
        self.R = sym.eye(2, 2)#*self.env.dt*self.env.N
        self.vars = [th1, th2, u1, u2, T1, T2]

    def check_linearizer(self, nsteps, actions=[0, 0]):
        vars = self.vars.copy()
        env = self.env
        for i in range(nsteps):
            sub_list = [(vars[0], env.th[0]),  # setting th1
                        (vars[1], env.th[1]),  # setting th2
                        (vars[2], env.u[0]),  # setting u1
                        (vars[3], env.u[1]),  # setting u2
                        (vars[4], actions[0]),  # setting T1
                        (vars[5], actions[1])]  # setting T2
            A_val = np.array(self.A.subs(sub_list))
            B_val = np.array(self.B.subs(sub_list))

            def dX(X, t):
                #np.reshape(X, (-1, 1))
                return  A_val.dot(X) + B_val.dot(actions)

            ttt = dX(np.array([env.th[0], env.th[1], env.u[0], env.u[1]]), [0])

            X_lin = odeint(dX, np.array([env.th[0], env.th[1], env.u[0], env.u[1]]), [1])[0]
            X, _, _, _ = self.env.step(actions)

            print("State: th1={},th2={}".format(X[0], X[1]))
            print("linearized States = th1={},th2={}".format(X_lin[0], X_lin[1]))

    def compute_lqr(self, old_actions, discrete=True):
        """
        Compute the discrete-time LQR controller.
        """
        vars = self.vars.copy()
        env = self.env
        sub_list = [(vars[0], env.th[0]), # setting th1
                    (vars[1], env.th[1]), # setting th2
                    (vars[2], env.u[0]),  # setting u1
                    (vars[3], env.u[1]),  # setting u2
                    (vars[4], old_actions[0]),  # setting T1
                    (vars[5], old_actions[1])] # setting T2
        A_val = self.A.subs(sub_list) * self.env.dt + np.eye(4)
        B_val = self.B.subs(sub_list) * self.env.dt
        a = np.array(A_val).astype(np.float64)
        b = np.array(B_val).astype(np.float64)
        r = np.array(self.R).astype(np.float64)
        q = np.array(self.Q).astype(np.float64)
        a, b, q, r = map(np.atleast_2d, (a, b, q, r))

        #print(a)
        # LQR gain
        if not discrete:
            p = scipy.linalg.solve_continuous_are(a, b, q, r)
            k = np.linalg.solve(r, b.T.dot(p))
        else:
            # k = (b.T * p * b + r)^-1 * (b.T * p * a)
            p = scipy.linalg.solve_discrete_are(a, b, q, r)
            bp = b.T.dot(p)
            tmp1 = bp.dot(b)
            tmp1 += r
            tmp2 = bp.dot(a)
            k = np.linalg.solve(tmp1, tmp2)
        return k


# def angle_normalize(x):
#     return (((x+np.pi) % (2*np.pi)) - np.pi)
#
#
# def angle_from_vertical(x):
#     x1 = angle_normalize(x + np.pi)
#     x2 = angle_normalize(x - np.pi)
#     return min(x1, x2)


env = gym.make("Pendulum2-v0", th_0=[np.pi*0.3, np.pi*0.8], u_0 = [-5., 0.], c=0, max_torque=100, dt=0.01, m=0.1)
env.reset()

#env.render(skip=50, read_file=True)
controller = ControllerPendulum2(env, method=1)
#controller.check_linearizer(100, actions=[10,10])
N = 500
actions = [0, 0]

for i in tqdm.trange(N):
    th, _, _, _ = env.step(actions)
    K = controller.compute_lqr(actions, discrete=True)
    state = np.array([env.th[0], env.th[1], env.u[0], env.u[1]])
    actions = -K.dot(state)
    actions = np.clip(actions, -env.max_torque, env.max_torque)
    if i%100 == 0:
        print(actions)

env.save_history()
env.render(skip=1)
# env.render(skip=10)
#
#
# vars = controller.vars.copy()
# sub_list = [(vars[0], np.pi), # setting th1
#             (vars[1], 0), # setting th2
#             (vars[2], 0),  # setting u1
#             (vars[3], 0),  # setting u2
#             (vars[4], 0),  # setting T1
#             (vars[5], 0)] # setting T2
#
# Q = sym.eye(4, 4)
# Q[0, 0] = 0.001
# Q[1, 1] = 0.001
# R = sym.eye(2,2)*0.01
#
# Bval = controller.B.subs(sub_list)
# Aval = controller.A.subs(sub_list)
# a = np.array(Aval).astype(np.float64)
# b = np.array(Bval).astype(np.float64)
# r = np.array(R).astype(np.float64)
# q = np.array(Q).astype(np.float64)
# a, b, q, r = map(np.atleast_2d, (a, b, q, r))
# p = scipy.linalg.solve_discrete_are(a, b, q, r)
# k = np.linalg.solve(r+b.T.dot(p).dot(b), b.T.dot(p).dot(a))
# print("A=", Aval)
# print("B=", Bval)
# print("Q=", q)
# print("r=", r)
# print("p=", p)
# print("k=", k)