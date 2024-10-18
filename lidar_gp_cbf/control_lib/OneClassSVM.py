import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class DynamicKernelSafetySVM:
    def __init__(self, grid_size_plot, kernel='rbf', nu=0.1, gamma='scale', N=360):
        self.reset_data()
        self.grid_size = grid_size_plot
        self.N = N
        # One-Class SVM for unsupervised learning of the safety function
        self.svm_model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)  # Use RBF or other kernels
        self.scaler = StandardScaler()  # For scaling input data
        self.X = None  # Placeholder for data

    def reset_data(self):
        self.data_X = []
        self.data_Y = []  # Not needed for One-Class SVM
        self.k = 0  # Reset iteration number

    def set_new_data(self, new_X):
        """
        Dynamically update the safety function with new data points.
        In One-Class SVM, no explicit labels are needed. The model learns the decision boundary
        between "safe" and "unsafe" dynamically from the data distribution.
        """
        # Add new data point to the dataset
        self.data_X.append(new_X)

        # Stack all the data points so far
        buffer_X = np.vstack(self.data_X)

        if len(self.data_X) >= self.N:  # Train only if we have more or equal amount than designated amount of datapoints
            # Scale the data points
            self.scaler.fit(buffer_X)
            X_scaled = self.scaler.transform(buffer_X)

            # Train the One-Class SVM with the accumulated data
            self.svm_model.fit(X_scaled)

    def predict_safety(self, t):
        """
        Predict whether a given point is "safe" or "unsafe" using the trained One-Class SVM.
        Returns:
        - 1 if the point is inside the decision boundary (considered safe)
        - -1 if the point is outside (considered unsafe)
        """
        t_scaled = self.scaler.transform([t])  # Scale the test point
        prediction = self.svm_model.predict(t_scaled)  # Predict safety

        return prediction

    def decision_function(self, t):
        """
        Return the decision function value for a given point `t`.
        The decision function value indicates how far a point is from the learned decision boundary.
        """
        t_scaled = self.scaler.transform([t])  # Scale the test point
        return self.svm_model.decision_function(t_scaled)

    '''................ Mapping the safety prediction .....................'''
    def __create_mesh_grid(self, field_x, field_y):
        m = int((field_x[1] - field_x[0]) // self.grid_size)
        n = int((field_y[1] - field_y[0]) // self.grid_size)
        gx, gy = np.meshgrid(np.linspace(field_x[0], field_x[1], m),
                             np.linspace(field_y[0], field_y[1], n))
        return gx.flatten(), gy.flatten()

    def draw_svm_whole_map_prediction(self, ax, field_x, field_y, robot_pos, sensing_rad, color='r'):
        """ Draw the safety map based on One-Class SVM's decision function. """
        data_point_x, data_point_y = self.__create_mesh_grid(field_x, field_y)
        grid_points = np.vstack((data_point_x, data_point_y)).T

        # Scale the grid points for prediction
        grid_points_scaled = self.scaler.transform(grid_points)

        # Predict the safety for each point on the grid
        safety_predictions = self.svm_model.decision_function(grid_points_scaled)

        # Reshape to match the grid dimensions
        safety_map = safety_predictions.reshape((len(np.unique(data_point_x)), len(np.unique(data_point_y))))

        # Plot the safety map
        ax.imshow(safety_map, extent=(field_x[0], field_x[1], field_y[0], field_y[1]), origin='lower', cmap=RdGn)
        ax.scatter(robot_pos[0], robot_pos[1], color=color, label="Robot Position")
        ax.set_title("Safety Prediction Map")