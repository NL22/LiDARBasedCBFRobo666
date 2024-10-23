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

def label_function(distance, sense_dist, safe_offset):
    """
    Returns the label based on the distance:
    - For distance >= sense_dist, return +1 (safe).
    - For distance <= safe_offset, return -1 (unsafe).
    - For distances between safe_offset and sense_dist, smoothly transition from +1 to -1.
    """
    
    steepness = 20 # Adjust steepness to control how sharp the transition is
    if distance <= safe_offset:
        return -1.0  # Assign -1 to points that are closer than safe_offset
    elif distance >= sense_dist:
        return 1.0  # Assign +1 to points beyond sense_dist
    else:
        # Smooth transition between +1 and -1 for points between safe_offset and sense_dist
        label = 2 / (1 + np.exp(-steepness * (distance - safe_offset))) - 1
        #label =(2 * (distance/sense_dist))-1
        return label

# Distance transformer
class DistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_point=np.array([0, 0])):  # Default reference point (origin)
        self.reference_point = reference_point

    def fit(self, X, y=None):
        return self  

    def transform(self, X):
        # Calculate the Euclidean distance to the reference point
        position_data = X[:, :2]
        distance = np.linalg.norm(position_data - self.reference_point, axis=1).reshape(-1, 1)
        return distance

'''________________________ GP Reg _________________________________________'''

class OnlineSVMModel():
    def __init__(self, kernel_type='rbf', C=1, epsilon=0.1, min_d_sample=0.1, iter_mem=50, grid_size_plot=0.1, dh_dt=0.01):
        self.reset_data()
        self.min_d_sample = min_d_sample
        self.mem_num = iter_mem
        self.grid_size = grid_size_plot
        self.dh_dt = dh_dt
        self.safe_offset = 0.5
        # Initialize the SVR model with appropriate parameters
        self.svm_model = SVR(kernel=kernel_type, C=1, epsilon=0.05, gamma=1)

        reference_point = np.array([0, 0])  # You can change this to any reference point
        self.distance_transformer = DistanceTransformer(reference_point=reference_point)
        self.scaler = StandardScaler()  # Add a scaler for feature scaling
        self.regr = make_pipeline(self.distance_transformer,self.scaler, self.svm_model) # Add a pipeline for ease of use
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
        self.k += 1
        if self.N != 0:
            forgetting_iters = list(range(self.k - self.mem_num, max(1, self.k - 2 * self.mem_num) - 1, -1))
            mask = ~np.isin(self.iter, forgetting_iters)

            if mask.shape[0] == self.data_X.shape[0] and mask.shape[0] == self.data_Y.shape[0]:
                print(f"Retaining {np.sum(mask)} points out of {len(mask)}")  # Add this line to see what's retained
                self.iter = self.iter[mask]
                self.data_Y = self.data_Y[mask]
                self.data_X = self.data_X[mask]
            else:
                #print(f"Mask shape: {mask.shape}, data_X shape: {self.data_X.shape}, data_Y shape: {self.data_Y.shape}")
                raise ValueError("Mismatch between mask and data sizes.")
            # If after filtering, data becomes empty, reset the dataset
            if len(self.data_X) == 0:
                self.data_X = None
                self.data_Y = None
                self.iter = None
                self.N = 0


    def set_new_data(self, new_X, new_Y=np.array([np.float64(1)]), sense_dist=np.inf, rob_pos=np.array([0,0,0]), safe_offset=0.05):
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
        theta = np.arctan2(new_X[0,1], new_X[0,0])  # Compute angle in radians
    
        # Create the new feature vector including x, y, and theta
        new_feature = np.array([[new_X[0,0], new_X[0,1], theta]])
        if self.data_X is None:
            # Initialize with the new data
            distance = np.linalg.norm(new_X)  # Calculate distance from the obstacle
            label = label_function(distance, sense_dist, safe_offset)
            self.data_X = new_feature
            self.data_Y = np.array([label])  # Assign label based on distance
            self.iter = np.array([self.k])  # Initialize iteration tracking array
        else:
            dis_to_mem = np.linalg.norm(self.data_X[:, 0:2] - new_X[0:2], axis=1)
            if min(dis_to_mem) > self.min_d_sample:

                # Calculate distance from robot to obstacle
                distance = np.linalg.norm(new_X)
                label = label_function(distance, sense_dist, safe_offset)
                
                # DEBUG: Output to check assigned label
                print(f"Distance: {distance}, Assigned Label: {label}")

                # Update the data arrays
                self.data_X = np.append(self.data_X, new_feature, axis=0)
                self.data_Y = np.append(self.data_Y, label)
                self.iter = np.append(self.iter, self.k)
                # Calculate a negative point
                direction_vec = new_X # Vector from robot to obstacle
                distance = np.linalg.norm(direction_vec)  # Distance between robot and obstacle
                neg_point = np.array([[0,0,theta]])
                if distance > 0:
                    # Calculate the safe point by moving 'safe_offset' units away from the robot
                    neg_point[0,:2] = direction_vec * safe_offset
                # Add a negative point to achieve a decent gradient
                self.data_X = np.append(self.data_X, neg_point, axis=0)
                self.data_Y = np.append(self.data_Y, -1.0)
                self.iter = np.append(self.iter, self.k)
                self.N = len(self.data_X)
            else:
                arg = np.argmin(dis_to_mem)
                self.iter[arg] = self.k

    def update_SVG(self):
        # Perform batch learning
        if len(self.data_X) > 3:  # Ensure there are enough points before training
            # Fit scaler and SVM model 
            #distances = self.distance_transformer.transform(self.data_X)
            #self.data_Y = self.y_scaler.fit_transform(self.data_Y.reshape(-1, 1)).ravel()  # Scale data_Y
            self.regr.fit(self.data_X, self.data_Y)
                

    def get_h_value(self, t):
        """
        Get the actual value of h(x) at the given point t using the SVM model.
        """
        #distances = self.distance_transformer.transform(t)
        
        return self.regr.predict(t)

    def get_cbf_safety_prediction(self, t):
        """
        Compute the safety prediction using the SVM, returning both the value and the gradient of h.
        
        Parameters:
        t : np.array
            The input state position(x,y) where we want to compute the safety prediction.
        
        
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
        # Get the actual value of h
        hsvm_xq = np.array([[np.float64(1)]])
        
        hsvm_xq[0,0] = self.get_h_value(t)[0]
        
        # Compute the gradient of RBF kernel with respect to the input t and kernel width sigma
        svm_G = self.compute_gradient(t,sigma = 2.0)
        # Compute the adjusted h value for safety constraints
        svm_h = hsvm_xq - self.dh_dt
        
        return svm_G, svm_h, hsvm_xq
    
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
        t = self.distance_transformer.transform(t)
        grad = np.zeros_like(t)
        self.plot_safety_boundary()
        # Loop over all support vectors
        for i in range(len(support_vectors)):
            t_i = support_vectors[i]  # Support vector
            alpha_i = dual_coef[0, i]  # Dual coefficient for this support vector
            
            # Compute the kernel between t and the support vector t_i
            K_xi_x = np.exp(-np.linalg.norm(t - t_i)**2 / (2 * sigma**2))
            
            # Compute the gradient of the RBF kernel
            grad += alpha_i * (- (t - t_i) / sigma**2) * K_xi_x
        
        return grad

    def plot_gradient_field(self, field_x, field_y):
        """
        Plot the gradient field of the safety function h over the specified area.
        
        field_x: tuple specifying the range of x-axis (min_x, max_x)
        field_y: tuple specifying the range of y-axis (min_y, max_y)
        """
        plt.figure()
        
        # Create a grid of points in the specified area
        xx, yy = np.meshgrid(np.linspace(field_x[0], field_x[1], 30), np.linspace(field_y[0], field_y[1], 30))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Initialize arrays to hold gradient components
        grad_x = np.zeros_like(xx)
        grad_y = np.zeros_like(yy)
        self.plot_safety_boundary
        # Compute gradients at each point on the grid
        for i in range(grid_points.shape[0]):
            point = grid_points[i].reshape(1, -1)  # Get a point (x, y)
            grad = self.compute_gradient_numerical(point, epsilon=1e-3)  # Compute gradient at this point
            grad_x.ravel()[i] = grad[0, 0]  # Gradient in x-direction
            grad_y.ravel()[i] = grad[0, 1]  # Gradient in y-direction
        
        # Plot the gradient field using quiver (vector field)
        plt.quiver(xx, yy, grad_x, grad_y)  # Arrow vectors represent gradients
        plt.title('Gradient Field of Safety Function')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

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
        """
        Plot the safety function's boundary, where h(x, y) crosses the threshold.
        """
        x_min, x_max = self.data_X[:, 0].min() - 0.1, self.data_X[:, 0].max() + 0.1
        y_min, y_max = self.data_X[:, 1].min() - 0.1, self.data_X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        h_values = self.get_h_value(grid_points)  # Evaluate h at all points on the grid
        h_values = h_values.reshape(xx.shape)
        
        # Plot the safety function as a contour map
        plt.contourf(xx, yy, h_values, levels=[-1, 0, 1], cmap='RdYlGn', alpha=0.5)
        plt.scatter(self.data_X[:, 0], self.data_X[:, 1], c=self.data_Y, cmap='RdYlGn', edgecolors='k')
        plt.colorbar()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Safety Function Boundary with X, Y Coordinates")
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
        if self.N>9: # Assign with 
            map_local=robot_pos[:2] - map_to_plot
            _,_,self.hpg_map=self.get_cbf_safety_prediction(map_local)
            x_plt_glob=robot_pos[0]+self.data_X[:,0]
            y_plt_glob=robot_pos[1]+self.data_X[:,1]
            self._pl_dataset.set_data(robot_pos[0]+self.data_X[:,0],robot_pos[1]+self.data_X[:,1])
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