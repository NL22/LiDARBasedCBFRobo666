import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import warnings

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

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


# Distance transformer
class DistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_point=np.array([0, 0])):  # Default reference point (origin)
        self.reference_point = reference_point

    def fit(self, X, y=None):
        return self  

    def transform(self, X):
        # Calculate the Euclidean distance to the reference point
        scaling_factor = 100
        distance = np.linalg.norm(X - self.reference_point, axis=1).reshape(-1, 1)
        return distance

'''________________________ GP Reg _________________________________________'''

class OnlineSVMModel():
    def __init__(self, kernel_type='rbf', C=1.0, epsilon=0.1, min_d_sample=0.1, iter_mem=50, grid_size_plot=0.1, dh_dt=0.01):
        self.reset_data()
        self.min_d_sample = min_d_sample
        self.mem_num = iter_mem
        self.grid_size = grid_size_plot
        self.dh_dt = dh_dt
        self.safe_offset = 0.5
        # Initialize the SVR model with appropriate parameters
        self.svm_model = SVR(kernel=kernel_type, C=1, epsilon=0.1, gamma=0.5)
        self.scaler = StandardScaler()  # Add a scaler for feature scaling
        self.regr = make_pipeline(self.scaler, self.svm_model) # Add a pipeline for ease of use
        self.initial_fit = False
        self.__prediction_plot=None
        self.init = False
        self.init_map=True
        self.set = False


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

    def set_new_data(self, new_X, new_Y=np.array([[-1.0]]), dist=np.inf,sense_dist=np.inf,rob_pos=np.array([0,0,0]), safe_offset=0.2):
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
            self.data_Y = new_Y   # Initialize empty array for Y
            self.iter = np.array([self.k])     # Initialize iteration tracking array

        # Check the distance of the new data point
        else:
            dis_to_mem = np.linalg.norm(self.data_X[:, 0:2] - new_X[0:2], axis=1)
            #if min(dis_to_mem) > self.min_d_sample:
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

                        # 2. Safe point (Far from obstacle)
                        #safe_point = new_X + direction_vec * safe_offset  # Move safe_offset units away from the obstacle
                        #self.data_X = np.append(self.data_X, safe_point, axis=0)
                        #self.data_Y = np.append(self.data_Y, +1.0)  # Label as safe
                        # self.iter = np.append(self.iter, self.k)
                        # 2. Generate multiple safe points around the obstacle in a fixed pattern
                        num_safe_points = 1  # You can adjust the number of safe points as needed
                        angles = np.linspace(0, 2 * np.pi, num_safe_points, endpoint=False)

                        # Create safe points uniformly distributed around the obstacle
                        for angle in angles:
                            safe_point = new_X - safe_offset * np.array([np.cos(angle), np.sin(angle)])  # Fixed offset circle
                            self.data_X = np.append(self.data_X, safe_point, axis=0)
                            self.data_Y = np.append(self.data_Y, +1.0)  # Label as safe
                            self.iter = np.append(self.iter, self.k)
            self.N = len(self.data_X)
            #else:
                # Update label if a new data point was detected nearby
                #arg = np.argmin(dis_to_mem)
                #self.iter[arg] = self.k

    def update_SVG(self):
        # Perform batch learning
        if len(self.data_X) > 3:  # Ensure there are enough points before training
                # Fit scaler and SVM model 
                self.regr.fit(self.data_X, self.data_Y)
                
        

    def get_h_value(self, t):
        """
        Get the actual value of h(x) at the given point t using the SVM model.
        """
        #distances = self.distance_transformer.transform(t)
        
        return self.regr.predict(t)

    def compute_gradient(self, t, sigma):
        """
        Computes gradient of RBF kernel utilizing SVM's weights

        Parameters:
        t : np.array
            Input state position(x,y) where we want to compute the safety prediction.
        sigma : float
            Defines the width of the RBF kernel
        
        Return:
        grad : np.array
            the gradients of RBF kernel wrt to position(x,y)
        """
        # Get the weifhts from the SVM model
        support_vectors = self.svm_model.support_vectors_
        dual_coef = self.svm_model.dual_coef_ 
        # Initialize the gradient vector
        grad = np.zeros_like(t)
        #self.plot_safety_boundary()
        # Loop over all support vectors
        for i in range(len(support_vectors)):
            t_i = support_vectors[i]  # Support vector
            alpha_i = dual_coef[0, i]  # Dual coefficient for this support vector
            
            # Compute the kernel between t and the support vector t_i
            K_xi_x = np.exp(-np.linalg.norm(t - t_i)**2 / (2 * sigma**2))
            
            # Compute the gradient of the RBF kernel
            grad += alpha_i * (- (t - t_i) / sigma**2) * K_xi_x
        
        return grad

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
        self.update_SVG()
        n = t.shape[0]  # Number of input points (rows in t)
    
        # Initialize arrays to store h values and gradients
        hsvm_xq = np.zeros((n, 1))  # Array for h values
        svm_h = np.zeros((n, 1))    # Array for h values minus dh_dt
        svm_G = np.zeros((n, t.shape[1]))  # Array for gradients, with the same dimensionality as t
        
        # Loop over all input points in t
        for i in range(n):
            # Get the actual value of h for the current input t[i]
            h_value = self.get_h_value(t[i].reshape(1, -1))[0]
            hsvm_xq[i, 0] = h_value  # Store the h value
            
            # Compute the numerical gradient of h with respect to t[i]
            gradient = self.compute_gradient(t[i].reshape(1, -1), sigma=1.0)
            svm_G[i, :] = gradient  # Store the gradient
            
            # Compute the adjusted h value for safety constraints
            svm_h[i, 0] = h_value - self.dh_dt
        
        return svm_G, svm_h, hsvm_xq
    
    def plot_h_wrt_distance(self):
        plt.figure()
    
        distances = np.linspace(0, 1, 100).reshape(-1, 1)
        h_values = self.regr.predict(distances)
        
        # Plot the relationship between distance and safety function h
        plt.plot(distances, h_values)
        plt.xlabel("Distanceeeeeeeeee")
        plt.ylabel("Safety Function heeeeeeeeeee")
        plt.title("Safety Function h vs Distanceeeeeeeeeeeeeee")
        
        # Show the new figure
        plt.show()

    def plot_safety_boundary(self):
        # Create a new figure window explicitly
        plt.figure()  # This ensures a new window is created for this plot

        # Create a meshgrid for the 2D space
        x_min, x_max = self.data_X[:, 0].min() - 0.1, self.data_X[:, 0].max() + 0.1
        y_min, y_max = self.data_X[:, 1].min() - 0.1, self.data_X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        # Predict the safety function h on the grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        h_values = self.regr.predict(grid_points)
        h_values = h_values.reshape(xx.shape)

        # Plot the decision boundary and data points
        plt.contourf(xx, yy, h_values, levels=[-1, 0, 1], cmap=RdGn, alpha=0.5)
        plt.scatter(self.data_X[:, 0], self.data_X[:, 1], c=self.data_Y, cmap=RdGn, edgecolors='k')
        plt.colorbar()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Safety Function Boundary with X, Y Coordinates")
        
        # Show the plot in the new figure window
        plt.show()

    def main_kernel(self,a, b,c,d, hypers):
        """
        kernel BETWEEN inputs a,b  
        kernel hyperparamters:       
        ell is the characteristic length scale
        sigma_f is the signal scale
        sigma_y is the observation noise
        hypers = np.array([l_gp, sigma_f, sigma_y])
        to ensure positivity, use full float range """
        """fixed hyper params"""
        [l_gp, sigma_f, sigma_y] = hypers
        # square exponential kernel
        SE = (lambda a, b,e: np.exp(-np.abs(e)*self.decay_rate)*np.exp( np.dot(a-b, a-b) / (-2.0*l_gp**2) ))
        #SE = (lambda a, b,e: np.exp( np.dot(a-b, a-b) / (-2.0*l_gp**2) ))
        kSE = kernel(SE)
        kSEab = kSE(a, b,c,d)
        
        # noise observation kernel
        kNOab= value_eye_fill(1,kSEab.shape)
        # main kernel is the sum of all kernels
        kab = ( sigma_f ** 2 * kSEab + sigma_y**2 * kNOab )
        return kab
                

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
        if self.N>9: # Assign with data
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