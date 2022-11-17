# overhead

import logging
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils

logger = logging.getLogger(__name__)

#*********************************************************************************************************

# environment parameters

FRAME_TIME = 1.0  # time interval
GRAVITY_ACCEL = 9.81/1000  # gravity constant
BOOST_ACCEL = 14.715/1000  # thrust constant, from class announcement

#*********************************************************************************************************

# system dynamics

class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):  

       """
       action[0] = thrust controller
       action[1] = omega controller
       state[0] = x
       state[1] = x_dot
       state[2] = y
        state[3] = y_dot
        state[4] = theta
        """
        # Apply gravity
        # Note: Here gravity is used to change velocity which is the second element of the state vector
        # Normally, we would do x[1] = x[1] + gravity * delta_time
        # but this is not allowed in PyTorch since it overwrites one variable (x[1]) that is part of the computational graph to be differentiated.
        # Therefore, I define a tensor dx = [0., gravity * delta_time], and do x = x + dx. This is allowed...
       delta_state_gravity = t.tensor([0., 0., 0., -GRAVITY_ACCEL * FRAME_TIME, 0.])
        # Thrust
        # Note: Same reason as above. Need a 5-by-1 tensor.
       N = len(state)
       state_tensor = t.zeros((N, 5))
       state_tensor[:, 1] = -t.sin(state[:, 4])
       state_tensor[:, 3] = t.cos(state[:, 4])
       delta_state = BOOST_ACCEL * FRAME_TIME * t.mul(state_tensor, action[:, 0].reshape(-1, 1))
        # Theta
       delta_state_theta = FRAME_TIME * t.mul(t.tensor([0., 0., 0., 0, -1.]), action[:, 1].reshape(-1, 1))
       state = state + delta_state + delta_state_gravity + delta_state_theta
        # Update state
       step_mat = t.tensor([[1., FRAME_TIME, 0., 0., 0.],
                                [0., 1., 0., 0., 0.],
                                [0., 0., 1., FRAME_TIME, 0.],
                                [0., 0., 0., 1., 0.],
                                [0., 0., 0., 0., 1.]])
       state = t.matmul(step_mat, state.T)


       return state
    

#*********************************************************************************************************

# a deterministic controller
# Note:
# 0. You only need to change the network architecture in "__init__"
# 1. nn.Sigmoid outputs values from 0 to 1, nn.Tanh from -1 to 1
# 2. You have all the freedom to make the network wider (by increasing "dim_hidden") or deeper (by adding more lines to nn.Sequential)
# 3. Always start with something simple

class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):    #customize
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            # You can add more layers here ************************************************************************** edit, don't fully understand this?
            nn.Sigmoid()
        )

    def forward(self, state):
        action = self.network(state)
        return action

#*********************************************************************************************************

# the simulator that rolls out x(1), x(2), ..., x(T)
# Note:
# 0. Need to change "initialize_state" to optimize the controller over a distribution of initial states
# 1. self.action_trajectory and self.state_trajectory stores the action and state trajectories along time

class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():   
        state = [1., 1., 0., 0., 0.]  # set the initial states ******************** [x, y, x_dot, y_dot, theta] make sure the orientation and the y-velocity match directions
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        return state[0]**2 + state[1]**2 + state[2]**2 + state[3]**2 + state[4]**2

#*********************************************************************************************************

# set up the optimizer
# Note:
# 0. LBFGS is a good choice if you don't have a large batch size (i.e., a lot of initial states to consider simultaneously)
# 1. You can also try SGD and other momentum-based methods implemented in PyTorch
# 2. You will need to customize "visualize"
# 3. loss.backward is where the gradient is calculated (d_loss/d_variables)
# 4. self.optimizer.step(closure) is where gradient descent is done

class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01) #set learning rate here *********************************************************

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        self.optimizer.step(closure)
        return closure()
    
    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))
            self.visualize()

    def visualize(self):       #customize: add graph or line for every state and add labels ******************************************************
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        # labeling data [x, y, x_dot, y_dot, theta]
        x = data[:, 0]
        y = data[:, 1]
        x_dot = data[:, 2]
        y_dot = data[:, 3]
        theta = data[:, 4]
        plt.plot(y, y_dot)
        plt.show()
        
        
#*********************************************************************************************************


# Now it's time to run the code!

T = 20  # number of time steps
dim_input = 5  # state space dimensions
dim_hidden = 60  # latent dimensions
dim_output = 2  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(5)  # solve the optimization problem