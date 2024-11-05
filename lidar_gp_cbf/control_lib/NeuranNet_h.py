import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils as nn_utils
import torch.nn.functional as F

from collections import deque

PYSIM = True
#PYSIM = False # for experiment or running via ROS

if PYSIM:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

warnings.filterwarnings('ignore')
'''________________________ color map ______________________________________'''
#red to green color map for safe and unsafe areas 
cdict = {'green':  ((0.0, 0.0, 0.0),   # no red at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

        'red': ((0.0, 1, 1),   # set to 0.8 so its not too bright at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0)),  # no green at 1

        'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
                  (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                  (1.0, 0.0, 0.0))   # no blue at 1
       }
RdGn = colors.LinearSegmentedColormap('GnRd', cdict)
'''________________________ functions ______________________________________'''

def kernel(f):      
        return lambda a, b,c,d: np.array(
        [[np.float64(f(a[i], b[j],c[i]-d[j]))  for j in range(b.shape[0])]
        for i in range(a.shape[0])] )

def value_eye_fill(value,shape):
    result = np.zeros(shape)
    np.fill_diagonal(result, value)
    return result

# Calculate the relative error
def relative_error(A,x,B):
    # Check the accuracy of the solution
    # Calculate the residual (error)
    residual = np.dot(A, x) - B 
    re= np.linalg.norm(residual) / np.linalg.norm(B)
    
    if re < 1e-6:
        accurate=True
    else:
        accurate=False
    return accurate
    

'''________________________ GP Reg _________________________________________'''

class SafetyNet():
    def __init__(self, min_d_sample=0.1, iter_mem=50, grid_size_plot=0.1, dh_dt=0.01, smoothing_window = 5):
        self.reset_data()
        self.min_d_sample = min_d_sample
        self.mem_num = iter_mem
        self.grid_size = grid_size_plot
        self.dh_dt = dh_dt
        self.safe_offset = 0.5
        # Initialize the neural network model with appropriate parameters
        input_dim = 2  # Dimensionality of the state space
        hidden_dim1 = 256
        hidden_dim2 = 128
        hidden_dim3 = 128
        super(SafetyNet, self).__init__()
        # Define the architecture of the neural network
        self.model = self.construct_net(input_dim = input_dim, hidden_dim1 = hidden_dim1, hidden_dim2 = hidden_dim2, hidden_dim3 = hidden_dim3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.initial_fit = False
        self.__prediction_plot=None
        self.init = False
        self.init_map=True
        self.set = False
        self.batch_size = 360
        self.smoothing_window = smoothing_window
        # Initialize deque to store recent safety function values for smoothing
        self.h_values = deque(maxlen=self.smoothing_window)
        self.learn_threat = False
       
    def construct_net(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, drop_prob = 0.2):
        """
        Function to construct the neural network architecture.
        Returns:
            model: A PyTorch model representing the network.
        """
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(drop_prob),  # Dropout after first layer
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(drop_prob),  # Dropout after second layer
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim3, 1),  # Single output for safety value (h)
        )
        
        # Initialize weights
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        return model
    
    def forward(self, x):
        """
        Forward pass for the model.
        """
        return self.model(x)

    def train_online(self, dataloader, epochs=7, max_grad_norm = 0.3, L = 1.0):
        """
        Perform online training using mini-batch SGD.
        Input: dataloader - DataLoader object containing training data
        """
        self.model.train()  # Set model to training mode
        for epoch in range(epochs):
            running_loss = 0.0
            for states_batch, safety_values_batch in dataloader:
                self.optimizer.zero_grad()
                # Forward pass
                predictions = self.forward(states_batch)
                # Compute main loss
                loss = self.criterion(predictions, safety_values_batch)
                
                # Compute Lipschitz penalty
                total_loss = loss

                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                #nn_utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimization step
                self.optimizer.step()
                running_loss += total_loss.item()
            avg_loss = running_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        self.model.eval()
        
    
    def compute_gradient(self, state, robot_ang):
        """
        Compute the gradient of the safety function w.r.t. the input state.
        Input: state - The state for which to compute the gradient
        Output: gradient - Gradient of the safety function (dx, dy, ...)
        """
        # Denormalize the state values
        distance = state[0, 0] * 0.5  # Denormalize distance
        angle = state[0, 1] * 2 * np.pi - np.pi  # Denormalize angle from [0, 1] to [-π, π]
        
        # Prepare the normalized input tensor for gradient computation
        normalized_input_tensor = torch.tensor(state, requires_grad=True)

        # Forward pass to compute h(d, θ)
        h_value = self.forward(normalized_input_tensor)
        
        # Backward pass to compute the gradients with respect to normalized distance and angle
        h_value.backward()
        
        # Retrieve the polar gradients (∂h/∂d and ∂h/∂θ) in the normalized space
        grad_r_norm = normalized_input_tensor.grad[0, 0].item()  # ∂h/∂distance_norm
        grad_theta_norm = normalized_input_tensor.grad[0, 1].item()  # ∂h/∂angle_norm
        
        # Convert gradients to denormalized form
        grad_r = grad_r_norm * 0.5 # Scale gradient for distance
        grad_theta = grad_theta_norm * (2*np.pi)   # Scale gradient for angle
        
        # Convert polar gradients to Cartesian gradients
        grad_x = grad_r * np.cos(robot_ang +angle) - grad_theta * (np.sin(robot_ang + angle)/distance)
        grad_y = grad_r * np.sin(robot_ang +angle) + grad_theta * (np.cos(robot_ang + angle)/distance)
        
        # Combine Cartesian gradients into a single vector
        gradient_cartesian = np.array([grad_x, grad_y]).reshape((1, 2))        
        
        return gradient_cartesian, h_value.item()


    def data_and_train(self):
        dataloader = DataLoader(TensorDataset(torch.FloatTensor(self.data_X), torch.FloatTensor(self.data_Y)), batch_size=len(self.data_X), shuffle=True)
    
        loss = self.train_online(dataloader)
            #print(f"Batch size: {self.data_X}, Epoch: {epoch+1}, Loss: {loss}")
        

    def reset_data(self):
        self.data_X = None
        self.data_Y = None
        self.N = 0
        self.k = 0
        self.iter = None
        self.iter_mem = None
        
    def new_iter(self):
        #update iteration number
        self.k+=1
        #go through recorded data and remove data that are older than iteration memory
        if (self.N!=0):
            forgetting_iters=list(range(self.k - self.mem_num, max(1, self.k - 2 * self.mem_num) - 1, -1))
            mask = ~np.isin(self.iter, forgetting_iters)
            # Ensure that the mask size matches with data_X and data_Y size
            if mask.shape[0] == self.data_X.shape[0] and mask.shape[0] == self.data_Y.shape[0]:
                # Apply the mask to retain only relevant data
                self.iter = self.iter[mask]
                self.data_Y = self.data_Y[mask]
                self.data_X = self.data_X[mask]

            # If after filtering, data becomes empty, reset the dataset
            if len(self.data_X) == 0:
                self.data_X = None
                self.data_Y = None
                self.iter = None
                self.N = 0


    # Modify to construct Tensor for training dataset
    def set_new_data(self, new_X, new_Y=np.array([np.float64(1)]), theta=0.0, safe_offset=0.05, sense_dist = 0.5):
        """
        Update the SVM model with new sensor data, ensuring a minimum sampling distance.
        
        Parameters:
        new_X : np.array
            The newly sensed position(s) (shape: 1x2 or more).
        new_Y : np.array
            The corresponding label(s), default is unsafe.
        sense_dist : float
            The sensing distance, used to determine if a point is 'inf' or not.
        safe_offset : float
            The offset to generate a safe point from the measured unsafe point.
        """

        # Create the new feature vector including x, y, and theta
        new_feature = np.array([new_X, theta[0]])
        distance = new_X 
        
        if self.data_X is None:
            # Initialize with the new data
            if(distance>=sense_dist):
                label = 1.0
            else:
                # We want a range from 1 to -1 
                # norm distance gives us  0 to 1 so we multiply by 2 and the -1 
                #label = 2 * (np.exp(-5 * (1 - new_X.item())) - 0.5)
                label = 2*(new_X.item()) - 1
            #self.data_X = new_feature
            self.data_X = np.array([[new_X.item(), theta[0]]])
            self.data_Y = np.array([[label]]) # Assign label based on distance
            self.iter = np.array([self.k])  # Initialize iteration tracking array
        else:
                #label = label_function(distance, sense_dist, safe_offset)

                # DEBUG: Output to check assigned label
                #print(f"Distance: {distance}, Assigned Label: {label}")

                # Update the data arrays
                #self.data_X = np.append(self.data_X, new_feature, axis=0)
                if(distance>=1.0):
                    self.data_X = np.append(self.data_X, [[new_X.item(), theta[0]]], axis=0)
                    self.data_Y = np.append(self.data_Y, [[+1.0]])
                    self.iter = np.append(self.iter, self.k)
                else:
                    #label = 2 * (np.exp(-5 * (1 - new_X.item())) - 0.5)
                    label = 2*(new_X.item()) - 1
                    self.data_X = np.append(self.data_X, [[new_X.item(), theta[0]]], axis=0)
                    self.data_Y = np.append(self.data_Y, [[label]])
                    self.iter = np.append(self.iter, self.k)
                # Add a negative point to achieve a decent gradient
                self.N = len(self.data_X)

    def get_smoothed_h_value(self, h_current):
        """
        Add the current h value to the deque and return the smoothed value.
        """
        self.h_values.append(h_current)
        return np.mean(self.h_values)  # Moving average

    def get_h_value(self, t):
        """
        Get the actual value of h(x) at the given point t using the model.
        Applies temporal smoothing.
        """
        # Predict safety function value for input state t
        h_current = self.forward(torch.FloatTensor(t)).item()
        
        # Smooth the output using a moving average
        smoothed_h_value = self.get_smoothed_h_value(h_current)
        return smoothed_h_value

    # A function ot get the collection of gradients for our bot 
    # Multi gradient circle around or bot
    # we can generate these readings for the circle of radius r(could be sensing dist)
    def evaluate_combined_gradient2(self, points ,robot_pos = 1.0, num_points=30):
        """
        Evaluate a combined safety gradient for multiple directions around the robot.
        
        Parameters:
        - network: Trained safety function model.
        - robot_pos: The robot's current [distance, theta] position.
        - num_points: Number of points around the robot for gradient evaluation.
        
        Returns:
        - combined_gradient: The resulting direction for increased safety.
        """
        # Generate angles and distances once for all points
        delta_theta = 2 * np.pi / num_points
        distances = np.full(num_points, robot_pos)
        thetas = np.arange(num_points) * delta_theta

        # Normalize angles only once
        normalized_thetas = (thetas + np.pi) / (2 * np.pi)

        # Stack distances and normalized_thetas as a batch (num_points, 2)
        #inputs = torch.FloatTensor(points)
        inputs = torch.FloatTensor(np.column_stack((distances, normalized_thetas)))

        # Compute gradients in batch
        inputs.requires_grad = True
        predicted_safety_values = self.forward(inputs)
        predicted_safety_values.backward(torch.ones_like(predicted_safety_values))

        # Extract gradients in batch
        grad_dists = inputs.grad[:, 0].numpy()
        grad_thetas = inputs.grad[:, 1].numpy()

        # Convert gradients from polar to Cartesian in batch
        grad_x = grad_dists * np.cos(thetas) - grad_thetas * np.sin(thetas)
        grad_y = grad_dists * np.sin(thetas) + grad_thetas * np.cos(thetas)

        # Combine gradients using weights based on magnitudes
        gradient_vectors = np.column_stack((grad_x, grad_y))
        magnitudes = np.linalg.norm(gradient_vectors, axis=1)
        weighted_gradient = np.average(gradient_vectors, axis=0, weights=magnitudes)
        combined_gradient = weighted_gradient / (np.linalg.norm(weighted_gradient) + 1e-8)

        return combined_gradient
    
    def evaluate_combined_gradient(self, robot_pos=1.0, num_points=30):
        """
        Evaluate a combined safety gradient for multiple directions around the robot.
        
        Parameters:
        - robot_pos: The robot's current [distance, theta] position.
        - num_points: Number of points around the robot for gradient evaluation.
        
        Returns:
        - combined_gradient: The resulting direction for increased safety.
        """
        # Generate angles and distances once for all points
        delta_theta = 2 * np.pi / num_points
        distances = np.full(num_points, robot_pos)
        thetas = np.arange(num_points) * delta_theta

        # Normalize angles only once
        normalized_thetas = (thetas + np.pi) / (2 * np.pi)

        # Stack distances and normalized_thetas as a batch (num_points, 2)
        inputs = torch.FloatTensor(np.column_stack((distances, normalized_thetas)))

        # Compute gradients in batch
        inputs.requires_grad = True
        predicted_safety_values = self.forward(inputs)
        predicted_safety_values.backward(torch.ones_like(predicted_safety_values))

        # Extract gradients in batch
        grad_dists = inputs.grad[:, 0].numpy()
        grad_thetas = inputs.grad[:, 1].numpy()

        # Convert gradients from polar to Cartesian in batch
        grad_x = grad_dists * np.cos(thetas) - grad_thetas * np.sin(thetas)
        grad_y = grad_dists * np.sin(thetas) + grad_thetas * np.cos(thetas)

        # Combine gradients using weights based on magnitudes
        gradient_vectors = np.column_stack((grad_x, grad_y))
        magnitudes = np.linalg.norm(gradient_vectors, axis=1)
        weighted_gradient = np.average(gradient_vectors, axis=0, weights=magnitudes)
        combined_gradient = weighted_gradient / (np.linalg.norm(weighted_gradient) + 1e-8)

        return combined_gradient

    def get_cbf_safety_prediction(self, t, compute_controll=0, robot_angle = [0.0]):
        """
        Compute the safety prediction using the SVM, returning both the value and the gradient of h.
        
        Parameters:
        t : np.array
            The input state (position, etc.) where we want to compute the safety prediction.
        
        
        Returns:
        gp_G : np.array
            The gradient of the function h(x).
        gp_h : np.array
            The value of the function h(x) minus dh_dt.
        hgp_xq : np.array
            The value of the function h(x).
        """
        # Learn from the collected data
        #self.data_and_train()
        n = t.shape[0]  # Number of input points (rows in t)

        # Initialize arrays to store h values and gradients
        hsvm_xq = np.zeros((n, 1))  # Array for h values
        svm_h = np.zeros((n, 1))    # Array for h values minus dh_dt
        svm_G = np.zeros((n, t.shape[1]))  # Array for gradients, with the same dimensionality as t
        # Make t into some form of distance 
        # Cast a distane forward and normalize?
        # Loop over all input points in t
        
        for i in range(n):
            # Get the actual value of h for the current input t[i]
            h_value = self.forward(torch.FloatTensor(t[i].reshape(1, -1)))
            hsvm_xq[i, 0] = h_value  # Store the h value
            
            # Compute the numerical gradient of h with respect to t[i]
            if(compute_controll):
                combined_gradient = self.evaluate_combined_gradient2(t[i].reshape(1, -1))
                svm_G[i, :] = combined_gradient    
            else:
                grad_cart, safety_h= self.compute_gradient(torch.FloatTensor(t[i].reshape(1, -1)), robot_ang= robot_angle[0])
                svm_G[i, :] = grad_cart # Store the gradient
                
            
            # Compute the adjusted h value for safety constraints
            svm_h[i, 0] = h_value - self.dh_dt
        # Compute the numerical gradient of h with respect to t[i]
        #if(compute_controll):
        #    combined_gradient = self.evaluate_combined_gradient2(t)
        #    svm_G[i, :] = combined_gradient  
        return svm_G, svm_h, hsvm_xq
                

    """................ Mapping the safety prediction..................... """
    
    def __create_mesh_grid(self, field_x, field_y):
        aa=0
        m = int( (field_x[1]+aa - field_x[0]-aa) //self.grid_size ) 
        n = int( (field_y[1]+aa - field_y[0]-aa) //self.grid_size ) 
        gx, gy = np.meshgrid(np.linspace(field_x[0]-aa, field_x[1]+aa, m), np.linspace(field_y[0]-aa, field_y[1]+aa, n))
        return gx.flatten(), gy.flatten()

    def draw_gp_whole_map_prediction(self, ax, field_x, field_y, ic, robot_pos, robot_theta, sensing_rad, color='r'):
        if self.init_map:
            """ initializing the mapping """
            data_point_x, data_point_y = self.__create_mesh_grid(field_x, field_y)
            r_x=data_point_x.shape[0]
            r_y=data_point_y.shape[0]
            self.t_map=np.append(np.reshape(data_point_x,(1,r_x)).T,np.reshape(data_point_y,(1,r_y)).T, axis=1)
            self.init_map=False
            
            # Assign handler, data will be updated later
            self._pl_dataset, = ax.plot(robot_pos[0], robot_pos[1], '.', color=color)

            circle_linspace = np.linspace(0., 2*np.pi, num=360, endpoint=False)
            self.def_circle = np.transpose(np.array([np.cos(circle_linspace), np.sin(circle_linspace)]))
        
        self.h_val_toplot = np.ones(self.t_map.shape[0])
        # Grab the closest data
        is_computed = np.linalg.norm(robot_pos[:2] - self.t_map, axis=1) < sensing_rad*1.5
        map_to_plot = self.t_map[is_computed]

        """ updating the map """
        #print('iter',self.iter)
        #print('data',self.data_X)
        if self.learn_threat: # Assign with data
        #if self.N >9:
            # Convert points(world coord) to distance and polar coords
            x_obs_world = map_to_plot[:,0]
            y_obs_world = map_to_plot[:, 1]

            # Get the robot's current position and orientation
            x_robot = robot_pos[0]
            y_robot = robot_pos[1]
            theta_robot = robot_theta  # Assume theta is the robot's heading

            # Transform the obstacle's world coordinates to the robot's local frame
            dx = x_obs_world - x_robot
            dy = y_obs_world - y_robot
            x_obs_local = np.cos(theta_robot) * dx + np.sin(theta_robot) * dy
            y_obs_local = -np.sin(theta_robot) * dx + np.cos(theta_robot) * dy
            # Convert to polar coordinates
            distances = np.sqrt(x_obs_local**2 + y_obs_local**2)  # Distances from robot
            angles = np.arctan2(y_obs_local, x_obs_local)         # Angles in radians from robot's orientation
            normalized_dist_map = distances/sensing_rad
            normalized_theta_map = (angles + np.pi)/(2 * np.pi)
            # Create the [dist, theta] feed
            polar_feed = np.vstack((normalized_dist_map, normalized_theta_map)).T
            _,_,self.hpg_map=self.get_cbf_safety_prediction(polar_feed)

            distances_maps = self.data_X[:, 0] * sensing_rad  # De-normalize distances
            angles_maps = self.data_X[:, 1] * 2 * np.pi - np.pi  # De-normalize angles from [0,1] to [-π, π]

            # Convert from polar (r, θ) to Cartesian (x, y) in the robot's local frame
            x_local_maps = distances_maps * np.cos(angles_maps)
            y_local_maps = distances_maps * np.sin(angles_maps)

            # Get the robot's current position and orientation
            # x_robot, y_robot are the robot's world position, and theta_robot is its orientation in radians
            x_world_maps = x_robot + (np.cos(theta_robot) * x_local_maps - np.sin(theta_robot) * y_local_maps)
            y_world_maps = y_robot + (np.sin(theta_robot) * x_local_maps + np.cos(theta_robot) * y_local_maps)

            # Set the transformed world coordinates for plotting
            self._pl_dataset.set_data(x_world_maps, y_world_maps)

            #self._pl_dataset.set_data(self.data_X[:,0], self.data_X[:,1])
        else: # Reset map + assing current position to plot
            self.hpg_map=np.ones(len(map_to_plot))
            self._pl_dataset.set_data([robot_pos[0]], [robot_pos[1]])    

        self.h_val_toplot[is_computed] = self.hpg_map.T[0]
        # self.hpg_map,_,_,_,_=self.update_gp_computation(self.t_map)
        

        circle_data = np.array([robot_pos[:2]]) + (self.def_circle*sensing_rad)

        if self.__prediction_plot is not None:
            self.__prediction_plot.set_array(self.h_val_toplot)
            # update circle
            self._pl_circle.set_data(circle_data[:,0], circle_data[:,1])
        else:
            self.__prediction_plot = ax.tripcolor(self.t_map[:,0],self.t_map[:,1], self.h_val_toplot,
                    vmin = -3, vmax = 3, shading='gouraud', cmap=RdGn)
            if PYSIM:
                axins1 = inset_axes(ax, width="25%", height="2%", loc='lower right')
                plt.colorbar(self.__prediction_plot, cax=axins1, orientation='horizontal', ticks=[-1,0,1])
                axins1.xaxis.set_ticks_position("top")
                # draw circle first time
                self._pl_circle, = ax.plot(circle_data[:,0], circle_data[:,1], '--', color='gray')
