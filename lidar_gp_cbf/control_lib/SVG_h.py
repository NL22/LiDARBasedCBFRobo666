import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import warnings

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

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

class OnlineSVMModel():
    def __init__(self, kernel_type='rbf', alpha=0.01, epsilon=0.1, min_d_sample=0.1, iter_mem=50, grid_size_plot=0.1, dh_dt=0.01):
        self.reset_data()
        self.min_d_sample = min_d_sample
        self.mem_num = iter_mem
        self.grid_size = grid_size_plot
        self.dh_dt = dh_dt
        self.safe_offset = 0.5
        # Initialize the SGD model with epsilon-insensitive loss for regression (SVM-like behavior)
        self.svm_model = SGDRegressor(loss="epsilon_insensitive", alpha=alpha, epsilon=epsilon, learning_rate='constant', eta0=0.01)
        self.scaler = StandardScaler()  # Add a scaler for feature scaling
        self.initial_fit = False  # To check if the model has been fitted at least once


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
            self.iter= self.iter[mask]
            self.data_Y= self.data_Y[mask]
            self.data_X= self.data_X[mask]
            #if data set is empty 
            if len(self.data_X)==0:
                self.data_X=None
                self.data_Y= None
                self.iter=None
                self.N=0

    def set_new_data(self, new_X, new_Y=np.array([[-2]]), sense_dist=np.inf, safe_offset=0.05):
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
            self.data_X = np.empty((0, 2))  # Initialize empty array for X
            self.data_Y = np.empty((0,))    # Initialize empty array for Y
            self.iter = np.empty((0,))      # Initialize iteration tracking array

        # Check distance of datapoint
        else:
            # Check distance of datapoint
            dis_to_mem=np.linalg.norm(self.data_X[:,0:2]-new_X[0:2], axis=1)
            if min( dis_to_mem)> self.min_d_sample :
                # Process new data point
                distance = np.linalg.norm(x)  # Assume new_X contains the distance or position
                if distance >= sense_dist:  # No obstacle detected (infinite distance)
                    # Safe point, label as +1
                    y = 1
                    self.data_X = np.append(self.data_X, [x], axis=0)
                    self.data_Y = np.append(self.data_Y, y)
                    self.iter = np.append(self.iter, self.k)
                else:
                    # Obstacle detected, two points are generated: one unsafe and one safe
                    # 1. Unsafe point (obstacle detected)
                    self.data_X = np.append(self.data_X, [x], axis=0)
                    self.data_Y = np.append(self.data_Y, -1)
                    self.iter = np.append(self.iter, self.k)

                    # 2. Safe point (just before the obstacle by safe_offset)
                    safe_point = x * ((distance - safe_offset) / distance)
                    self.data_X = np.append(self.data_X, [safe_point], axis=0)
                    self.data_Y = np.append(self.data_Y, 1)
                    self.iter = np.append(self.iter, self.k)
                self.N=len(self.data_X)
            else:
                #update label because a new data was detected near by
                arg=np.argmin(dis_to_mem)
                self.iter[arg]=self.k

    def update_SVG(self):
        # Perform online learning using partial_fit after scaling
        if len(self.data_X) > 5:  # Ensure there are enough points before training
            if not self.initial_fit:
                # Fit scaler and SVM model for the first time
                self.scaler.fit(self.data_X)
                self.data_X = self.scaler.transform(self.data_X)
                self.svm_model.partial_fit(self.data_X, self.data_Y)
                self.initial_fit = True
            else:
                # Scale new data and fit incrementally
                new_X_scaled = self.scaler.transform(new_X)
                self.svm_model.partial_fit(new_X_scaled, new_Y.flatten())


    def get_h_value(self, t):
        """
        Get the actual value of h(x) at the given point t using the SVM model.
        """
        return self.svm_model.predict(t)

    def compute_gradient_numerical(self, t, epsilon=1e-5):
        """
        Compute the numerical gradient of h(x) with respect to x at point t.
        
        Parameters:
        t : np.array
            The point where we want to compute the gradient.
        epsilon : float
            Small value to compute the finite difference.
        
        Returns:
        gradient : np.array
            The gradient of the function h(x) at point t.
        """
        t = np.array(t)
        grad = np.zeros_like(t)
        
        # Iterate through each dimension of t
        for i in range(t.shape[1]):
            t_plus = np.copy(t)
            t_minus = np.copy(t)
            
            # Slightly perturb the current dimension
            t_plus[0, i] += epsilon
            t_minus[0, i] -= epsilon
            
            # Compute the finite difference for this dimension
            h_plus = self.svm_model.predict(self.scaler.transform(t_plus))  # Scale inputs
            h_minus = self.svm_model.predict(self.scaler.transform(t_minus))  # Scale inputs
                
            grad[0, i] = (h_plus - h_minus) / (2 * epsilon)
        
        return grad 

    def get_cbf_safety_prediction(self, t, dis_f):
        """
        Compute the safety prediction using the SVM, returning both the value and the gradient of h.
        
        Parameters:
        t : np.array
            The input state (position, etc.) where we want to compute the safety prediction.
        dis_f : Unused
        
        Returns:
        gp_G : np.array
            The gradient of the function h(x).
        gp_h : np.array
            The value of the function h(x) minus dh_dt.
        hgp_xq : np.array
            The value of the function h(x).
        """
        # Learn from the collected data
        update_SVG()
        # Get the actual value of h
        hsvm_xq = self.get_h_value(t)
        
        # Compute the numerical gradient of h with respect to the input t
        svm_G = self.compute_gradient_numerical(t)
        
        # Compute the adjusted h value for safety constraints
        svm_h = hsvm_xq - self.dh_dt
        
        return svm_G, svm_h, hsvm_xq
    
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
        if self.N!=0: # Assign with data
            _,_,self.hpg_map=self.get_cbf_safety_prediction(map_to_plot)
            self._pl_dataset.set_data(self.data_X[:,0], self.data_X[:,1])
        else: # Reset map + assing current position to plot
            self.hpg_map=np.ones(len(map_to_plot))
            self._pl_dataset.set_data(robot_pos[0], robot_pos[1])    

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
