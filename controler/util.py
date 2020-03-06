import tqdm
import scipy
import numpy as np
import sympy as sym
from scipy.integrate import odeint

from lin_A_B_pend3 import *
from lin_A_B_airplane import *
class Controller:
    def __init__(self, env, use_sympy=False):
        self.env = env
        self.use_sympy = use_sympy
        if use_sympy:
            self.A, self.B = [None]*2 # These attributes will be initialized later.
            self.vars = None
            self._linerize_sympy()

    def _linerize_sympy(self):
        """
        This will load the symbolic form of matrices A and B
        """
        raise(NotImplementedError("implement _linerize_sympy function."))

    def linearize(self, current_state, old_action):
        """
        This function should return numpy arrays of A and B
        """
        raise(NotImplementedError("implement linearize function."))


    def check_linearizer(self, actions, nsteps):
        """
        This will compare linearized model output with
        the full non-linear output. Ideally, the outputs
        should be very close.
        """
        def dX(X, t):
            """ Linear model. """
            return A_val.dot(X) + B_val.dot(actions)

        env = self.env
        A_val, B_val = self.linearize()

        for _ in range(nsteps):
            X_lin = odeint(dX, np.array([env.x[0], env.x[1], env.x[2], env.x[3]]), [1])[0]
            X, _, _, _ = self.env.step(actions)
            print("State: th1={},th2={}".format(X[0], X[1]))
            print("linearized States: th1={},th2={}".format(X_lin[0], X_lin[1]))

    def lqr(self, Q, R, old_action, discrete=True):
        """
        Compute the discrete-time LQR controller.
        """
        A_val, B_val = self.linearize(self.env.x, old_action)
        a, b, q, r = map(np.atleast_2d, (A_val, B_val, Q, R))

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

    def ilqr(self, Q, R, Qf, num_steps, x_begin, x_goal, eps=0.001, u_init=None, itr_max=200):
        """
            Compute the iterative LQR (iLQR) controller.
        """
        itr_ilqr = 0
        n, m = Q.shape[0], R.shape[0]
        x_bar = np.zeros((n, num_steps + 1))
        x_bar[:, 0] = x_begin

        l = np.zeros((m, num_steps))
        L = np.zeros((m, n, num_steps))

        if u_init is None:
            u_bar = np.ones((m, num_steps))
        else:
            u_bar = u_init.copy()

        self.env.reset()
        self.env.x = x_bar[:, 0]
        for i in range(num_steps):
            self.env.x, _, _, _ = self.env.step(u_bar[:, i])
            x_bar[:, i + 1] = self.env.x

        print("\n start ilqr iterations")

        x_bar_old = x_bar.copy()

        while itr_ilqr < itr_max:
            itr_ilqr += 1
            qf = Qf.T.dot(x_bar[:, -1]) - Qf.T.dot(x_goal)
            P = Qf
            p = qf
            for i in reversed(range(num_steps)):
                q = Q.T.dot(x_bar[:, i]) - Q.T.dot(x_goal)
                r = R.T.dot(u_bar[:, i])

                lt, Lt, P, p = self._backward_riccati_recursion(P, Q, p, q, R, r, u_bar[:, i], x_bar[:, i])
                l[:, i] = lt
                L[:, :, i] = Lt

            u_bar_old = u_bar.copy()

            self.env.reset()
            self.env.x = x_bar[:, 0]
            for i in range(num_steps):
                d_u = l[:, i] + L[:, :, i].dot(x_bar[:, i] - x_bar_old[:, i])
                u_bar[:, i] += d_u
                self.env.x, _, _, _ = self.env.step(u_bar[:, i])
                x_bar[:, i + 1] = self.env.x

            x_bar_old = x_bar.copy()

            if np.linalg.norm(u_bar_old-u_bar) < eps:
                print("converged after {} trials with residual |u_bar_old-u_bar|={}".format(itr_ilqr, np.linalg.norm(u_bar_old-u_bar)))
                break
            else:
                print("itr #{}, |u_bar_old-u_bar|={}".format(itr_ilqr, np.linalg.norm(u_bar_old - u_bar)))
        return x_bar, u_bar, l, L

    def _backward_riccati_recursion(self, P, Q, p, q, R, r, u_bar_i, x_i):
        # A_val, B_val = self.linearize(self.env.x, u_bar_i)
        A_val, B_val = self.linearize(x_i, u_bar_i)
        Qxk  = q + A_val.T.dot(p)
        Quk  = r + B_val.T.dot(p)
        Qxxk = Q + (A_val.T.dot(P)).dot(A_val)
        Quuk = R + (B_val.T.dot(P)).dot(B_val)
        Quxk = 0 + (B_val.T.dot(P)).dot(A_val)

        l = -np.linalg.solve(Quuk, Quk)
        L = -np.linalg.solve(Quuk, Quxk)

        p = Qxk - (L.T.dot(Quuk)).dot(l)
        P = Qxxk - (L.T.dot(Quuk)).dot(L)

        return l, L, P, p

    def run_lqr(self, Q, R, x_goal, nsteps, old_action=None):
        if old_action is None:
            old_action = [0.] * (self.env.n_step // 2)
        all_actions = []
        for i in tqdm.trange(nsteps):
            K = self.lqr(Q, R, old_action)
            action = -K.dot(np.array(self.env.x) - x_goal)
            action = np.clip(action, -self.env.max_action, self.env.max_action)
            all_actions.append(action)
            if i % 25 == 0:
                print(action)
            self.env.x, _, _, _ = self.env.step(action)

        return all_actions

    def run_ilqr(self, Q, R, Qf, x_goal, nsteps, u_init=None):
        all_actions = []
        x_bar, u_bar, l, L = self.ilqr(Q, R, Qf, nsteps, self.env.x, x_goal, u_init=u_init)
        self.env.reset()
        for i in tqdm.trange(nsteps):
            action = u_bar[:, i] + l[:, i] + L[:, :, i].dot(self.env.x - x_bar[:, i])
            action = np.clip(action, -self.env.max_action, self.env.max_action)
            all_actions.append(action)
            if i%20 == 0:
                print("time step= %d, actions" %i, end="")
                with np.printoptions(precision=3, suppress=True, formatter={'float': '{: 0.3f}'.format}):
                    print(action)
            self.env.x, _, _, _ = self.env.step(action)

        return all_actions


class ControllerDoublePendulum(Controller):
    def __init__(self, env, use_sympy=False):
        super().__init__(env, use_sympy=use_sympy)

    def _linerize_sympy(self):
        """
        This will load the symbolic form of matrices A and B
        """
        th1, th2 = sym.symbols('th1, th2')
        u1, u2 = sym.symbols('u1, u2')
        T1, T2 = sym.symbols('T1, T2')

        M = sym.Matrix([[2, sym.cos(th2 - th1)],
                        [sym.cos(th2 - th1), 1]])
        C = sym.Matrix([[sym.sin(th2 - th1) * u2 ** 2],
                        [-sym.sin(th2 - th1) * u1 ** 2]])
        G = self.env.g / self.env.L * sym.Matrix([[sym.sin(th1) * 2],
                                                  [sym.sin(th2)]
                                                  ])
        T = 1. / (self.env.m * self.env.L ** 2) * sym.Matrix([[T1],
                                                              [T2]
                                                              ])
        F = -1. * self.env.c / (self.env.m * self.env.L ** 2) * sym.Matrix([[u1],
                                                                            [u2]
                                                                            ])
        aMat = M.inv() * (T + F + C + G)
        q0 = aMat.diff(th1)
        q1 = aMat.diff(th2)
        q2 = aMat.diff(u1)
        q3 = aMat.diff(u2)
        q4 = aMat.diff(T1)
        q5 = aMat.diff(T2)

        self.A = sym.Matrix([[0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [q0[0], q1[0], q2[0], q3[0]],
                             [q0[1], q1[1], q2[1], q3[1]]])

        self.B = sym.Matrix([[0, 0],
                             [0, 0],
                             [q4[0], q5[0]],
                             [q4[1], q5[1]]])
        self.vars = [th1, th2, u1, u2, T1, T2]

    def linearize(self, current_state, old_action):
        """
        This function should return numpy arrays of A and B
        """
        if self.use_sympy:
            vars = self.vars.copy()
            sub_list = [(vars[0], current_state[0]),  # setting th1
                        (vars[1], current_state[1]),  # setting th2
                        (vars[2], current_state[2]),  # setting u1
                        (vars[3], current_state[3]),  # setting u2
                        (vars[4], old_action[0]),  # setting T1
                        (vars[5], old_action[1])]  # setting T2
            A_val = self.A.subs(sub_list) * self.env.dt + np.eye(4)
            B_val = self.B.subs(sub_list) * self.env.dt

            A_val = np.array(A_val)
            B_val = np.array(B_val)

            A_val = A_val.astype(np.float64)
            B_val = B_val.astype(np.float64)
        else:
            th1, th2, u1, u2 = current_state
            c = self.env.c
            T1, T2 = old_action
            m, L, g = self.env.m, self.env.L, self.env.g
            dth1dd_dth1 = (2 * np.cos(th1 - th2) * np.sin(th1 - th2) * (
                    2 * c * u1 - 2 * T1 + T2 * np.cos(th1 - th2) - c * u2 * np.cos(th1 - th2) - 4 * L * g * m * np.sin(
                th1) + 2 * L ** 2 * m * u2 ** 2 * np.sin(th1 - th2) + L * g * m * np.cos(th1 - th2) * np.sin(
                th2) + L ** 2 * m * u1 ** 2 * np.cos(th1 - th2) * np.sin(th1 - th2))) / (
                                  L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4) ** 2) - (
                                  T2 * np.sin(th1 - th2) - c * u2 * np.sin(th1 - th2) + 4 * L * g * m * np.cos(
                              th1) - 2 * L ** 2 * m * u2 ** 2 * np.cos(th1 - th2) - L ** 2 * m * u1 ** 2 * np.cos(
                              th1 - th2) ** 2 + L ** 2 * m * u1 ** 2 * np.sin(th1 - th2) ** 2 + L * g * m * np.sin(
                              th1 - th2) * np.sin(th2)) / (
                                  L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4))

            dth2dd_dth1 = - (T1 * np.sin(th1 - th2) - c * u1 * np.sin(th1 - th2) + 2 * L ** 2 * m * u1 ** 2 * np.cos(
                th1 - th2) + L ** 2 * m * u2 ** 2 * np.cos(th1 - th2) ** 2 - L ** 2 * m * u2 ** 2 * np.sin(
                th1 - th2) ** 2 - 2 * L * g * m * np.cos(th1 - th2) * np.cos(th1) + 2 * L * g * m * np.sin(th1 - th2) * np.sin(
                th1)) / (
                                  L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4)) - (2 * np.cos(th1 - th2) * np.sin(th1 - th2) * (
                    2 * T2 - 2 * c * u2 - T1 * np.cos(th1 - th2) + c * u1 * np.cos(th1 - th2) + 2 * L * g * m * np.sin(
                th2) + 2 * L ** 2 * m * u1 ** 2 * np.sin(th1 - th2) - 2 * L * g * m * np.cos(th1 - th2) * np.sin(
                th1) + L ** 2 * m * u2 ** 2 * np.cos(th1 - th2) * np.sin(th1 - th2))) / (
                                  L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4) ** 2)

            dth1dd_dth2 = (T2 * np.sin(th1 - th2) - c * u2 * np.sin(th1 - th2) - 2 * L ** 2 * m * u2 ** 2 *np.cos(
                th1 - th2) - L ** 2 * m * u1 ** 2 * np.cos(th1 - th2) ** 2 + L ** 2 * m * u1 ** 2 * np.sin(
                th1 - th2) ** 2 + L * g * m * np.cos(th1 - th2) * np.cos(th2) + L * g * m * np.sin(th1 - th2) * np.sin(th2)) / (
                                  L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4)) - (2 * np.cos(th1 - th2) * np.sin(th1 - th2) * (
                    2 * c * u1 - 2 * T1 + T2 * np.cos(th1 - th2) - c * u2 * np.cos(th1 - th2) - 4 * L * g * m * np.sin(
                th1) + 2 * L ** 2 * m * u2 ** 2 * np.sin(th1 - th2) + L * g * m * np.cos(th1 - th2) * np.sin(
                th2) + L ** 2 * m * u1 ** 2 * np.cos(th1 - th2) * np.sin(th1 - th2))) / (
                                  L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4) ** 2)

            dth2dd_dth2 = (T1 * np.sin(th1 - th2) - c * u1 * np.sin(th1 - th2) - 2 * L * g * m * np.cos(
                th2) + 2 * L ** 2 * m * u1 ** 2 * np.cos(
                th1 - th2) + L ** 2 * m * u2 ** 2 * np.cos(th1 - th2) ** 2 - L ** 2 * m * u2 ** 2 * np.sin(
                th1 - th2) ** 2 + 2 * L * g * m * np.sin(th1 - th2) * np.sin(th1)) / (
                                      L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4)) + (
                                  2 * np.cos(th1 - th2) * np.sin(th1 - th2) * (
                                  2 * T2 - 2 * c * u2 - T1 * np.cos(th1 - th2) + c * u1 * np.cos(
                              th1 - th2) + 2 * L * g * m * np.sin(
                              th2) + 2 * L ** 2 * m * u1 ** 2 * np.sin(th1 - th2) - 2 * L * g * m * np.cos(th1 - th2) * np.sin(
                              th1) + L ** 2 * m * u2 ** 2 * np.cos(th1 - th2) * np.sin(th1 - th2))) / (
                                  L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4) ** 2)

            dth1dd_dthd1 = (2 * m * u1 * np.cos(th1 - th2) * np.sin(th1 - th2) * L ** 2 + 2 * c) / (
                L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4))

            dth2dd_dthd1 = -(4 * m * u1 * np.sin(th1 - th2) * L ** 2 + c * np.cos(th1 - th2)) / (
                L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4))

            dth1dd_dthd2 = -(- 4 * m * u2 * np.sin(th1 - th2) * L ** 2 + c * np.cos(th1 - th2)) / (
                L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4))

            dth2dd_dthd2 = (- 2 * m * u2 * np.cos(th1 - th2) * np.sin(th1 - th2) * L ** 2 + 2 * c) / (
                L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4))

            A_val = np.zeros((4, 4))
            A_val[0, 2] = 1.0
            A_val[1, 3] = 1.0
            A_val[2, 0] = dth1dd_dth1
            A_val[3, 0] = dth2dd_dth1
            A_val[2, 1] = dth1dd_dth2
            A_val[3, 1] = dth2dd_dth2
            A_val[2, 2] = dth1dd_dthd1
            A_val[3, 2] = dth2dd_dthd1
            A_val[2, 3] = dth1dd_dthd2
            A_val[3, 3] = dth2dd_dthd2
            A_val = A_val * self.env.dt + np.eye(4)

            B_val = np.zeros((4, 2))
            B_val[2, 0] = -2 / (L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4))
            B_val[2, 1] = np.cos(th1 - th2) / (L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4))
            B_val[3, 0] = np.cos(th1 - th2) / (L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4))
            B_val[3, 1] = -2 / (L ** 2 * m * (np.cos(th1 - th2) ** 2 - 4))
            B_val *= self.env.dt

        return A_val, B_val

class ControllerTriplePendulum(Controller):
    def __init__(self, env, use_sympy=False):
        super().__init__(env, use_sympy=use_sympy)

    def _linerize_sympy(self):
        """
        This will load the symbolic form of matrices A and B
        """
        th1, th2, th3 = sym.symbols('th1, th2, th3')
        u1, u2, u3 = sym.symbols('u1, u2, u3')
        T1, T2, T3 = sym.symbols('T1, T2, T3')

        M = sym.Matrix([[3.0, 2.0 * sym.cos(th1 - th2), 1.0 * sym.cos(th1 - th3)],
                        [2.0 * sym.cos(th1 - th2), 2.0, 1.0 * sym.cos(th2 - th3)],
                        [1.0 * sym.cos(th1 - th3), 1.0 * sym.cos(th2 - th3), 1.0]])
        C = sym.Matrix([[ 2.0 * sym.sin(th1 - th2) * u2 ** 2 + 1.0 * sym.sin(th1 - th3) * u3 ** 2],
                        [-2.0 * sym.sin(th1 - th2) * u1 ** 2 + 1.0 * sym.sin(th2 - th3) * u3 ** 2],
                        [-1.0 * sym.sin(th1 - th3) * u1 ** 2 - 1.0 * sym.sin(th1 - th3) * u3 ** 2]
                        ])
        C *= -1.0
        G = self.env.g / self.env.L * sym.Matrix([[sym.sin(th1) * 3],
                                                  [sym.sin(th2) * 2],
                                                  [sym.sin(th3) * 1]])
        T = 1. / (self.env.m * self.env.L ** 2) * sym.Matrix([[T1],
                                                              [T2],
                                                              [T3]])
        F = -1. * self.env.c / (self.env.m * self.env.L ** 2) * sym.Matrix([[u1],
                                                                            [u2],
                                                                            [u3]])
        aMat = M.inv() * (T + F + C + G)
        q0 = aMat.diff(th1)
        q1 = aMat.diff(th2)
        q2 = aMat.diff(th3)
        q3 = aMat.diff(u1)
        q4 = aMat.diff(u2)
        q5 = aMat.diff(u3)
        q6 = aMat.diff(T1)
        q7 = aMat.diff(T2)
        q8 = aMat.diff(T3)

        self.A = sym.Matrix([[0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 1., 0.],
                             [0., 0., 0., 0, 0., 1.],
                             [q0[0], q1[0], q2[0], q3[0], q4[0], q5[0]],
                             [q0[1], q1[1], q2[1], q3[1], q4[1], q5[1]],
                             [q0[2], q1[2], q2[2], q3[2], q4[2], q5[2]]
                             ])

        self.B = sym.Matrix([[0., 0., 0.],
                             [0., 0., 0.],
                             [0., 0., 0.],
                             [q6[0], q7[0], q8[0]],
                             [q6[1], q7[1], q8[1]],
                             [q6[2], q7[2], q8[2]]
                             ])

        self.vars = [th1, th2, th3, u1, u2, u3, T1, T2, T3]

    def linearize(self, current_state, old_action):
        """
        This function should return numpy arrays of A and B
        """
        if self.use_sympy:
            vars = self.vars.copy()
            sub_list = [(vars[0], current_state[0]),  # setting th1
                        (vars[1], current_state[1]),  # setting th2
                        (vars[2], current_state[2]),  # setting th3
                        (vars[3], current_state[3]),  # setting u1
                        (vars[4], current_state[4]),  # setting u2
                        (vars[5], current_state[5]),  # setting u3
                        (vars[6], old_action[0]),  # setting T1
                        (vars[7], old_action[1]),  # setting T1
                        (vars[8], old_action[2])]  # setting T3
            A_val = self.A.subs(sub_list)
            B_val = self.B.subs(sub_list)

            A_val = np.array(A_val) * self.env.dt + np.eye(6)
            B_val = np.array(B_val) * self.env.dt

            A_val = A_val.astype(np.float64)
            B_val = B_val.astype(np.float64)
        else:
            th1, th2, th3, u1, u2, u3 = current_state
            c = self.env.c
            T1, T2, T3 = old_action
            m, L, g = self.env.m, self.env.L, self.env.g

            A_val = np.zeros((6,6))
            A_val[0, 0] = A00(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[1, 0] = A10(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[2, 0] = A20(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[3, 0] = A30(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[4, 0] = A40(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[5, 0] = A50(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)

            A_val[0, 1] = A01(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[1, 1] = A11(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[2, 1] = A21(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[3, 1] = A31(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[4, 1] = A41(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[5, 1] = A51(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)

            A_val[0, 2] = A02(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[1, 2] = A12(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[2, 2] = A22(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[3, 2] = A32(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[4, 2] = A42(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[5, 2] = A52(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)

            A_val[0, 3] = A03(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[1, 3] = A13(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[2, 3] = A23(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[3, 3] = A33(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[4, 3] = A43(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[5, 3] = A53(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)

            A_val[0, 4] = A04(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[1, 4] = A14(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[2, 4] = A24(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[3, 4] = A34(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[4, 4] = A44(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[5, 4] = A54(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)

            A_val[0, 5] = A05(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[1, 5] = A15(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[2, 5] = A25(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[3, 5] = A35(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[4, 5] = A45(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            A_val[5, 5] = A55(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)

            A_val = A_val * self.env.dt + np.eye(6)

            B_val = np.zeros((6, 3))
            B_val[0, 0] = B00(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[1, 0] = B10(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[2, 0] = B20(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[3, 0] = B30(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[4, 0] = B40(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[5, 0] = B50(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)

            B_val[0, 1] = B01(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[1, 1] = B11(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[2, 1] = B21(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[3, 1] = B31(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[4, 1] = B41(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[5, 1] = B51(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)

            B_val[0, 2] = B02(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[1, 2] = B12(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[2, 2] = B22(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[3, 2] = B32(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[4, 2] = B42(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)
            B_val[5, 2] = B52(th1, th2, th3, u1, u2, u3, T1, T2, T3, m, g, L, c)

            B_val *= self.env.dt
        return A_val, B_val



class ControllerAirPlane(Controller):
    def __init__(self, env):
        super().__init__(env, use_sympy=False)

    def linearize(self, current_state, old_action):
        """
        This function should return numpy arrays of A and B
        """
        x, y, z, u, v, w, phi, theta, psi, p, q, r = current_state

        X, Y, Z, L, M, N = old_action
        Ix, Iy, Iz, Ixz, g, m = self.env.Ix, self.env.Iy, self.env.Iz, self.env.Ixz, self.env.g, self.env.m

        A_val = np.zeros((12, 12))
        for i in range(12):
            for j in range(12):
                # This for loop will call A_?_?.py function that determine the entries of A. The functions are stored
                # in lin_A_B_airplane.py and are generated with Matlab sybmolic operations.
                A_val[i, j] = globals()["A_%d_%d" %(i+1, j+1)](x, y, z, u, v, w, phi, theta, psi, p, q, r, Ix, Iy, Iz, Ixz, g, m)
        A_val = A_val * self.env.dt + np.eye(12)

        B_val = np.zeros((12, 6))
        for i in range(12):
            for j in range(6):
                B_val[i, j] = globals()["B_%d_%d" % (i+1, j+1)](X, Y, Z, L, M, N, Ix, Iy, Iz, Ixz, g, m)
        B_val *= self.env.dt

        return A_val, B_val


from gym.envs.registration import make
import matplotlib.pyplot as plt
if __name__ == "__main__":

    # env = make("Pendulum2-v0", x_0=[np.pi * 0.9, -np.pi * 0.8, 0., -1.], dt=0.01)
    # env.reset()
    # control = ControllerDoublePendulum(env)
    # n_step_lqr, n_step_ilqr = 250, 150
    # Q = np.eye(4, 4)
    # Q[1, 1] = 0
    # Q[2, 2] = 0
    # Qf = np.eye(4, 4) * 1000
    # R = np.eye(2, 2)
    # x_goal = [0., 0., 0., 0.]
    # ilqr_actions = control.run_ilqr(Q, R, Qf, x_goal, n_step_ilqr)
    # lqr_actions = control.run_lqr(Q, R, x_goal, n_step_lqr, ilqr_actions[-1])
    # print(env.x)
    # env.render()

    n_step_lqr, n_step_ilqr = 200, 50
    Q = np.eye(6, 6) *10
    Qf = np.eye(6, 6) * 1000
    R = np.eye(3, 3)
    x_goal = [0., 0., 0., 0., 0., 0.]

    x_0 = [ 5.396 , 0.368,  3.022, -8.616 , 2.409 , 8.418]
    x_mid = [1.5, 1., 1.5, 1., 1., 1.]
    x_mid2 = [0.5, 0.5, 0.5, 0.0, 0.0, -0.]
    env = make("Pendulum3-v0", dt=0.02, x_0=x_0)
    env.reset()

    u_init = np.zeros((3, n_step_ilqr))
    u_init[0, :n_step_ilqr] = np.linspace(1, 0, n_step_ilqr)
    control = ControllerTriplePendulum(env, use_sympy=False)
    ilqr_actions_1 = control.run_ilqr(Q, R, Qf, x_mid, n_step_ilqr, u_init=u_init)

    env.env.x_0 = env.x
    ilqr_actions_2 = control.run_ilqr(Q, R, Qf, x_mid2, n_step_ilqr)#, u_init=u_init*0.2)

    env.env.x_0 = env.x
    ilqr_actions_3 = control.run_ilqr(Q, R, Qf, x_goal, n_step_ilqr)  # , u_init=u_init*0.2)

    try:
        lqr_actions = control.run_lqr(Q, R, x_goal, n_step_lqr, ilqr_actions_3[-1])
    except:
        lqr_actions = [np.zeros(3)]*n_step_lqr

    env.env.x_0 = x_0
    env.reset()

    all_actions = ilqr_actions_1 + ilqr_actions_2 + ilqr_actions_3 + lqr_actions
    for i in range(n_step_ilqr + n_step_ilqr + n_step_ilqr + n_step_lqr):
        env.x, _, _, _ = env.step(all_actions[i])
    env.render()