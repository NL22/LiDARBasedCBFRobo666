import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from library.visualize_mobile_robot import sim_mobile_robot
from library.detect_obstacle import DetectObstacle
from library.ex6p3_obstacles import dict_obst_vertices

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 30 # total simulation duration in seconds
# Set initial state
init_state = np.array([-4., -3.5, 0]) # px, py, theta
#init_state = np.array([4., 0., np.pi*3/4]) # numpy array for goal / the desired [px, py, theta]
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-5, 5)
field_y = (-4, 4)

# Define sensor's sensing range and resolution
sensing_range = 1. # in meter
sensor_resolution = np.pi/8 # angle between sensor data in radian

R_s = 0.4 # m
robot_r = 0.21 # m
wheel_r = 0.1 # m
max_wheel_speed_rot = 20 # rad/s

# Control gains for linear and rotatiotal velocity
k_omega = 1.5
k_v = 0.7


# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_sensor_endpoint(robot_state, sensors_dist):
    # assuming sensor position is in the robot's center
    sens_N = round(2*np.pi/sensor_resolution)
    sensors_theta = [i*2*np.pi/sens_N for i in range(sens_N)]
    obst_points = np.zeros((3,sens_N))

    R_WB = np.array([ [np.cos(robot_state[2]), -np.sin(robot_state[2]), robot_state[0] ], \
        [np.sin(robot_state[2]),  np.cos(robot_state[2]), robot_state[1] ], [0, 0, 1] ])
    for i in range(sens_N):
        R_BS = np.array([ [np.cos(sensors_theta[i]), -np.sin(sensors_theta[i]), 0 ], \
            [np.sin(sensors_theta[i]),  np.cos(sensors_theta[i]), 0 ], [0, 0, 1] ])
        temp = R_WB @ R_BS @ np.array([sensors_dist[i], 0, 1])
        obst_points[:,i] = temp

    return obst_points[:2,:]


def compute_control_input(desired_state, robot_state, dist_reading, obst_points):
    # Compute go-to-goal control
    v = desired_state[:2] - robot_state[:2]
    theta = robot_state[2]
    theta_d = np.arctan2((desired_state[1] - robot_state[1]), (desired_state[0]- robot_state[0]))
    ang_diff = theta_d - theta
    # Limit angle difference between pi and -pi
    ang_diff = ( (ang_diff+ np.pi) % (2*np.pi) ) - np.pi
    omega = k_omega * ang_diff
    v_control = k_v*np.sqrt(v[0]**2 + v[1]**2)
    u_gtg = np.array([v_control, omega])
    v_control, omega, w_l, w_r = limit_wheel_speed(u_gtg[0], u_gtg[1])
    u_gtg = np.array([v_control, omega])

    obstacle_coords = []
    # Save closest points of the obstacle (if any detected by the sensor)
    ind = 0
    for reading in dist_reading:
        if(reading < 1):
            obstacle_coords.append(obst_points[:,ind])
        ind += 1
    # If no obstacles are detected by the sensor --> use go-to-goal
    if(len(obstacle_coords) == 0):
        current_input = u_gtg
    else:
        current_input = compute_QP(robot_state, u_gtg, obstacle_coords)

    return current_input, w_l, w_r

def compute_QP(robot_state, u_gtg, obstacle_coords):
    h = []
    H = np.zeros([len(obstacle_coords), 2])

    # Calculate h functions for each sensor measurement
    ind = 0
    # Calculate h and H for each point of the obstacle(s) detected
    for coord in obstacle_coords:
        h.append(compute_h(robot_state, coord))
        H[ind,:] = -2*(robot_state[:2] - coord)
        ind += 1
    b = np.empty([len(h)])
    # Gamma functions for elements in h
    ind = 0
    for elem in h:
        b[ind] = 0.1*elem
        ind += 1
    # Create cvxopt matrices for QP 
    Q = 2*cvxopt.matrix(np.eye(2), tc = 'd')
    c = -2 * cvxopt.matrix(u_gtg, tc = 'd')
    H = cvxopt.matrix(H, tc = 'd')
    b = cvxopt.matrix(b, tc = 'd')
    cvxopt.solvers.options["show_progress"] = False
    sol = cvxopt.solvers.qp(Q,c, H,b, verbose = True)
    u = np.array([sol['x'][0], sol['x'][1]])
    return u
    
def compute_h(robot_state, coord):
    dist = np.sqrt((robot_state[0] - coord[0])**2 + (robot_state[1] - coord[1])**2)
    h = dist**2 - R_s**2
    return h


def change_desired_state(desired_state, robot_state, ind):
    dist = np.linalg.norm(desired_state[:2] - robot_state[:2])
    if(dist <= 0.2):
        ind += 1
    return ind

def limit_wheel_speed(v_control, omega):
    # Calculate wheel rotational speeds: 
    w_r = (2*v_control + omega*2*robot_r) / (2*wheel_r)
    w_l = (2*v_control - omega*2*robot_r) / (2*wheel_r)
    # Check if desired wheel speeds exceed the limit 
    if(abs(w_r) > max_wheel_speed_rot and abs(w_l) > max_wheel_speed_rot):
        # Calculate the ratio between the wheels to achieve the same ration with limited rotational speeds
        ratio = w_r/w_l
        if(ratio > 1):
            w_r = w_r/abs(w_r) * max_wheel_speed_rot
            w_l = w_l/abs(w_l) * max_wheel_speed_rot / ratio
        elif(ratio < 1):
            w_l = w_l/abs(w_l) * max_wheel_speed_rot
            w_r = w_l/abs(w_l) * max_wheel_speed_rot * ratio
        else:
            w_r = w_r/abs(w_r) * max_wheel_speed_rot
            w_l = w_l/abs(w_l) * max_wheel_speed_rot

    elif(abs(w_l) > max_wheel_speed_rot):
         w_l = w_l/abs(w_l) * max_wheel_speed_rot
         
    elif(abs(w_r) > max_wheel_speed_rot):
        w_r = w_r/abs(w_r) * max_wheel_speed_rot

    # If wheel rotational speeds were not exceeded, return original values 
    else: 
        return v_control, omega, w_l, w_r
    # If one or both wheel speeds exceeded the limit, calculate new control values: 
    omega = ((w_r - w_l)*wheel_r)/(2*robot_r)
    v_control = ((w_r + w_l)*wheel_r)/2

    return v_control, omega, w_l, w_r




# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state1 = np.array([4., 0., 0.]) # numpy array for goal / the desired [px, py, theta]
    desired_state2 = np.array([-0.5, 3.7, 0.])
    desired_state3 = np.array([-4., -3.5, 0.])
    desired_states = np.array([desired_state1, desired_state2, desired_state3])
    ind = 0
    # Desired state is the first goal in the beginning
    desired_state = desired_states[ind]
    

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 2) ) # for [vlin, omega] vs iteration time
    wheel_rot_speed_history = np.zeros( (sim_iter, 2) )

    # Initiate the Obstacle Detection
    range_sensor = DetectObstacle( sensing_range, sensor_resolution)
    for key, value in dict_obst_vertices.items():
        range_sensor.register_obstacle_bounded( value )


    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        # sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state1)
        sim_visualizer.show_goal(desired_state2)
        sim_visualizer.show_goal(desired_state3)
        sim_visualizer.show_goal(desired_state3)

        # Display the obstacle
        for key, value in dict_obst_vertices.items():
            sim_visualizer.ax.plot(value[:,0], value[:,1], '--r')
        
        # get sensor reading
        # Index 0 is in front of the robot. 
        # Index 1 is the reading for 'sensor_resolution' away (counter-clockwise) from 0, and so on for later index
        distance_reading = range_sensor.get_sensing_data( robot_state[0], robot_state[1], robot_state[2])
        # compute and plot sensor reading endpoint
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)
        pl_sens, = sim_visualizer.ax.plot(obst_points[0], obst_points[1], '.') #, marker='X')
        pl_txt = [sim_visualizer.ax.text(obst_points[0,i], obst_points[1,i], str(i)) for i in range(len(distance_reading))]

    for it in range(sim_iter):
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Get information from sensors
        distance_reading = range_sensor.get_sensing_data( robot_state[0], robot_state[1], robot_state[2])
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)

        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input, w_l, w_r = compute_control_input(desired_state, robot_state, distance_reading, obst_points)
        #------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input
        wheel_rot_speed_history[it] = np.array([w_l, w_r])
        


        ind = change_desired_state(desired_state, robot_state, ind)
        desired_state = desired_states[ind]

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
            # update sensor visualization
            pl_sens.set_data(obst_points[0], obst_points[1])
            for i in range(len(distance_reading)): pl_txt[i].set_position((obst_points[0,i], obst_points[1,i]))
        
        #--------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of UNICYCLE model
        theta = robot_state[2]
        B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        robot_state = robot_state + Ts*(B @ current_input) # will be used in the next iteration
        # robot_state[2] = ( (robot_state[2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        #desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, wheel_rot_speed_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, wheel_rot_speed_history = simulate_control()


    # ADDITIONAL PLOTTING
    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:,0], label='v [m/s]')
    ax.plot(t, input_history[:,1], label='omega [rad/s]')
    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.title("Inputs")
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    ax.plot(t, state_history[:,0], label='px [m]')
    ax.plot(t, state_history[:,1], label='py [m]')
    ax.plot(t, state_history[:,2], label='theta [rad]')
    ax.plot(t, goal_history[:,0], ':', label='goal px [m]')
    ax.plot(t, goal_history[:,1], ':', label='goal py [m]')
    ax.plot(t, goal_history[:,2], ':', label='goal theta [rad]')
    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.title("States")
    plt.grid()

    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, wheel_rot_speed_history[:,0], label='w_l [rad/s]')
    ax.plot(t, wheel_rot_speed_history[:,1], label='w_r [rad/s]')
    ax.set(xlabel="t [s]", ylabel="Rotational velocities of the wheels")
    plt.legend()
    plt.title("Wheel rotational velocities")
    plt.grid()

    plt.show()