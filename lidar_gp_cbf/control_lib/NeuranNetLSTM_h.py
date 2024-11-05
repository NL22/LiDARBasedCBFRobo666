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

class LearnableSmoothingLayer(nn.Module):
    def __init__(self, kernel_size=3):
        super(LearnableSmoothingLayer, self).__init__()
        self.smoothing_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # Initialize kernel weights with uniform values for smoothing
        nn.init.constant_(self.smoothing_conv.weight, 1 / kernel_size)

    def forward(self, h_values):
        # Expect input shape to be (batch_size, 1, num_values) for 1D convolution
        if h_values.dim() == 2:
            h_values = h_values.unsqueeze(1)  # Add channel dimension
        smoothed_h = self.smoothing_conv(h_values)
        return smoothed_h.squeeze(1)  # Remove channel dimension after convolution

    

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
        hidden_dim1 = 64
        hidden_dim2 = 32
        self.lstm_hidden_dim = 128
        super(SafetyNet, self).__init__()

        # Define the architecture of the neural network with LSTM
        self.model = self.construct_net(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()
        self.initial_fit = False
        self.__prediction_plot = None
        self.init = False
        self.init_map = True
        self.set = False
        self.batch_size = 360
        self.smoothing_window = smoothing_window
        self.trained = False
        # Initialize deque to store recent safety function values for smoothing
        self.h_values = deque(maxlen=self.smoothing_window)
       
    def construct_net(self, input_dim, hidden_dim1, hidden_dim2, drop_prob = 0.2):
        """
        Function to construct the neural network architecture.
        Returns:
            model: A PyTorch model representing the network.
        """
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            #nn.Linear(hidden_dim2, self.lstm_hidden_dim)  # Prepare input for LSTM
            nn.Linear(hidden_dim2, 1),  # Single output for safety value (h)
        )

        # Initialize the LSTM separately and the final output layer
        #self.lstm = nn.LSTM(input_size=self.lstm_hidden_dim, hidden_size=self.lstm_hidden_dim, batch_first=True)
        #self.output_layer = nn.Sequential(
         #   nn.Linear(self.lstm_hidden_dim, 1),  # Single output for safety value (h)
        #    nn.Tanh()
        #)

        # Initialize weights
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        return model
    
    def forward(self, x):
        """
        Forward pass for the model.
        """
        x = self.model(x)
        
        # Add batch dimension if x is a single point (not a sequence)
        #if x.dim() == 2:
        #    x = x.unsqueeze(1)  # Add sequence dimension
        
        # Pass through LSTM
        #lstm_out, _ = self.lstm(x)
        
        # Use the last LSTM output for the final prediction
        #output = self.output_layer(lstm_out[:, -1, :])  # Last time step
        return x

    def train_online(self, dataloader, epochs=3, max_grad_norm = 1, L = 1.0):
        """
        Perform online training using mini-batch SGD.
        Input: dataloader - DataLoader object containing training data
        """
        self.model.train()
        #self.lstm.train()
        #self.output_layer.train()
        
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
                nn_utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimization step
                self.optimizer.step()
                running_loss += total_loss.item()
            avg_loss = running_loss / len(dataloader)
            print("Avergae Loss: " + str(avg_loss))
        self.model.eval()
    
    def compute_gradient(self, state, aggregation_method = 'sum'):
        """
        Compute the gradient of the safety function w.r.t. the input state.
        Input: state - The state for which to compute the gradient
        Output: gradient - Gradient of the safety function (dx, dy, ...)
        """
        self.model.eval()  # Set model to evaluation mode
        gradients = np.zeros((self.data_X.shape[0], self.data_X.shape[1])) 
        state_tensor = torch.FloatTensor(self.data_X)
        state_tensor.requires_grad = True
        # Forward pass
        predicted_safety_value = self.forward(state_tensor)
            
        gradient_vector = torch.ones_like(predicted_safety_value)
    
        # Backward pass to compute the gradient
        predicted_safety_value.backward(gradient=gradient_vector)
        
        # Extract and return the computed gradients
        gradients = state_tensor.grad.detach().numpy()
        
        # Aggregate gradients according to specified method
        if aggregation_method == 'average':
            aggregated_gradient = np.mean(gradients, axis=0)
        elif aggregation_method == 'sum':
            aggregated_gradient = np.sum(gradients, axis=0)
        else:
            raise ValueError("Invalid aggregation method. Choose either 'average' or 'sum'.")
        
        # Normalize the aggregated gradient for consistent direction
        #aggregated_gradient /= (np.linalg.norm(aggregated_gradient) + 1e-8)
        
        return aggregated_gradient


    def data_and_train(self):
        dataloader = DataLoader(TensorDataset(torch.FloatTensor(self.data_X), torch.FloatTensor(self.data_Y)), batch_size=len(self.data_X), shuffle=True)
        # Simulate 10 epochs for each batch size
        loss = self.train_online(dataloader)
            #print(f"Batch size: {self.data_X}, Epoch: {epoch+1}, Loss: {loss}")
        self.trained = True
       

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
    def set_new_data(self, new_X, new_Y=np.array([[-3.0]]), dist=np.inf,sense_dist=np.inf,rob_pos=np.array([0,0,0]), safe_offset=0.2):
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
        
        if self.data_X is None:
            # No previous data, so initialize with the new data
            self.data_X = new_X
            self.data_Y = +1.0   # Initialize empty array for Y
            self.iter = np.array([self.k])     # Initialize iteration tracking array

        # Check the distance of the new data point
        else:
            dis_to_mem = np.linalg.norm(self.data_X[:, 0:2] - new_X[0:2], axis=1)
            if min(dis_to_mem) > self.min_d_sample:
                # If the distance is larger than sensing distance, it is safe
                if dist >= sense_dist:
                        self.data_X = np.append(self.data_X, new_X, axis=0)
                        self.data_Y = np.append(self.data_Y, +1.0)
                        self.iter = np.append(self.iter, self.k)
                else:
                        # Obstacle detected, two points are generated: one unsafe and one safe

                        # Direction towards the point from roboto
                        direction_vec = rob_pos[:2] - new_X
                        distance = np.linalg.norm(direction_vec)
                        direction_vec = direction_vec / distance
                            
                        # 1. Unsafe point (Obstacle detected)
                        self.data_X = np.append(self.data_X, new_X, axis=0)
                        self.data_Y = np.append(self.data_Y, -1.0)  # Label as unsafe
                        self.iter = np.append(self.iter, self.k)

                        # 2. Generate multiple safe points around the obstacle in a fixed pattern
                        num_safe_points = 1  # You can adjust the number of safe points as needed
                        angles = np.linspace(0, 2 * np.pi, num_safe_points, endpoint=False)

                        # Create safe points uniformly distributed around the obstacle
                        #for angle in angles:
                        #    safe_point = new_X - safe_offset * np.array([np.cos(angle), np.sin(angle)])  # Fixed offset circle
                        #    self.data_X = np.append(self.data_X, safe_point, axis=0)
                        #    self.data_Y = np.append(self.data_Y, +1.0)  # Label as safe
                        #    self.iter = np.append(self.iter, self.k)
                self.N = len(self.data_X)
            else:
                # Update label if a new data point was detected nearby
                arg = np.argmin(dis_to_mem)
                self.iter[arg] = self.k

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

    def get_cbf_safety_prediction(self, t):
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
        
        n = t.shape[0]  # Number of input points (rows in t)

        # Initialize arrays to store h values and gradients
        hsvm_xq = np.zeros((n, 1))  # Array for h values
        svm_h = np.zeros((n, 1))    # Array for h values minus dh_dt
        svm_G = np.zeros((n, t.shape[1]))  # Array for gradients, with the same dimensionality as t
        
        # Loop over all input points in t
        for i in range(n):
            # Get the actual value of h for the current input t[i]
            h_value = self.forward(torch.FloatTensor(t[i].reshape(1, -1)))
            hsvm_xq[i, 0] = h_value  # Store the h value
            
            # Compute the numerical gradient of h with respect to t[i]
            gradient = self.compute_gradient(torch.FloatTensor(t[i].reshape(1, -1)))
            svm_G[i, :] = gradient  # Store the gradient
            
            # Compute the adjusted h value for safety constraints
            svm_h[i, 0] = h_value - self.dh_dt
        
        return svm_G, svm_h, hsvm_xq
                

    """................ Mapping the safety prediction..................... """
    
    def __create_mesh_grid(self, field_x, field_y):
        aa=0
        m = int( (field_x[1]+aa - field_x[0]-aa) //self.grid_size ) 
        n = int( (field_y[1]+aa - field_y[0]-aa) //self.grid_size ) 
        gx, gy = np.meshgrid(np.linspace(field_x[0]-aa, field_x[1]+aa, m), np.linspace(field_y[0]-aa, field_y[1]+aa, n))
        return gx.flatten(), gy.flatten()

    def draw_gp_whole_map_prediction(self, ax, field_x, field_y, ic, robot_pos, sensing_rad, color='r'):
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
        if self.trained: # Assign with data
            _,_,self.hpg_map=self.get_cbf_safety_prediction(map_to_plot)
            self._pl_dataset.set_data(self.data_X[:,0], self.data_X[:,1])
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
