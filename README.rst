# PrefsForDriving

The code is tested with python3.6 and python3.7 and julia 1.3.1
Here is the list of libraries with the version that is the code 
is tested on. 

we expect other versions of the above libraries (packages)
are also compatible with our program. The only exeception is
Flux.jl. You need to use 0.8.* version of this package, since
the newer version has some issues with auto-differentiation.

Also make sure you build PyCall package with that version
For this, you can use:

```
 using Pkg
 ENV["PYTHON"] = python_path # for example /usr/bin/python3.6
 Pkg.build(“PyCall”)
 Pkg.add(“PyCall”)
```

Please see https://github.com/JuliaPy/PyCall.jl for  more details. 

## necessary python libraries
numpy==1.17.1  
pyglet==1.4.10  
Theano==1.0.4  
matplotlib==3.1.3  


## necessary julia packages
Mamba==0.12.2  
Optim==0.20.0  
LinearAlgebra  
ForwardDiff==0.10.9  
LineSearches==7.0.1  
Distributions==0.22.3  
JLD2==0.1.11  
DelimitedFiles  
Flux==0.8.3  
BSON==0.2.5  
Plots==0.28.4  
PyCall==1.91.2  


## instructions

### 1- python
in two separate terminals run traj_A.py and traj_B.py. 
Both files are located in python_model. This will allow us 
to simulate two driving trajectories at the same time. 
```
cd PrefsForDriving/python_model
python3.6 traj_A.py
 ```

(in a different terminal)
```
cd PrefsForDriving/python_model
python3.6 traj_B.py	
```

### 2- julia
#### 2-1: generating train data
```
cd PrefsForDriving
julia
include("driving_simulate.jl")
reward_iteration(100)
```

#### 2-2: YOU MUST EXIT JULIA AND START OVER
```
exit()
```
 
#### 2-3: generate test data.
```
include("driving_simulate.jl")
@load "test_inputs_75.jld2"
respond_to_test_set(u_test)
```


# gym_new_env

## Intro

This repo includes:
1. OpenAI Gym environment with some additional environments including
    1.1 inverted double pendulum: two-link pendulum with point masses m at the end of each link. Links are massless. Viscous friction with a coefficient c is assumed. Both links are actuated. This is a four dimensional dynamical system. [th1, th2, dth1, dth2]. Parameters include:
        *  m: mass of point masses (default value = 0.5 kg)
        *  L: length of each link (default value = 0.5 m)
        *  c: viscous friction coefficient (default value = 0.1 N.s)
        *  g: gravitational acceleration (default value = 9.8 m/s^2)
        *  max_action: maximum torque on each link (default value = 1000 N.m)
        *  x_0: initial position (default value = [0.]*4)
        *  integration_method: integration scheme, options are "1st" which is a first order euler update and "2nd"
                                 which is crank-nicolson update. 
        *  dt: timestep (default value = 0.001 s)
    1.2 inverted triple pendulum: three-link pendulum. parameters are similar to inverted double pendulum. 
    1.3 airplane:
2. Conroller Implementation using LQR (linear–quadratic regulator) and ILQR (iterative linear–quadratic regulator) methods
3. Neural Network Controller: Training algorithm for designing neural network controller using behavior clonning.




## Installation
1. required libraries: pickle, numpy, scipy, matplotlib
2. From the top level directory, `gym_new_env`, run `pip install -e .`

## Usage
1. double pendulum
```
    env = make("Pendulum2-v0", dt=0.01)
    env.reset()
    for i in range(250):
        env.step([0., 0.]) # no torque
    env.render()
```   
2. controlling double pendulum using mixed ilqr and lqr:
    ```
    from gym.envs.registration import make
    from controler.util import ControllerDoublePendulum
    env = make("Pendulum2-v0", x_0=[1., 2., 0., -1.], dt=0.01) # initial position is [th1, th2, v1, v2]
    env.reset()
    control = ControllerDoublePendulum(env)
    n_step_lqr, n_step_ilqr = 250, 150
    Q = np.eye(4, 4)
    Q[1, 1] = 0
    Q[2, 2] = 0
    Qf = np.eye(4, 4) * 1000
    R = np.eye(2, 2)
    x_goal = [0., 0., 0., 0.]
    ilqr_actions = control.run_ilqr(Q, R, Qf, x_goal, n_step_ilqr)
    lqr_actions = control.run_lqr(Q, R, x_goal, n_step_lqr, ilqr_actions[-1])
    env.render() # or env.animate() to make a .gif file
    ```
3. controlling airplane using mixed ilqr and lqr with some intermediate way-points:
    ```
    import numpy as np
    from gym.envs.registration import make
    n_step_lqr, n_step_ilqr = 1500, 30
    Q = np.eye(12, 12) *10
    Qf = np.eye(12, 12) * 1000
    R = np.eye(6, 6)
    x_goal = [40., 0., 40., 0.,  0., 0., 0., .5, .5, 0., 0., 0.]
    x_0    = [0.,  0., 0.,  2.,  0., 0., .5, 0., 0., 0., 0., 0.]
    x_med  = [10., 0., 10., 0.,  0., 0., 0., 0., 0., 0., 0., 0.]
    env = make("AirPlane-v0", dt=0.01, x_0=x_0, g=1.)
    env.reset()
    from util import ControllerAirPlane
    control = ControllerAirPlane(env)
    ilqr_actions = control.run_ilqr(Q, R, Qf, x_med, n_step_ilqr)
    control.run_lqr(Q, R, x_goal, n_step_lqr, ilqr_actions[-1])
    env.render(skip=5)
    ```
