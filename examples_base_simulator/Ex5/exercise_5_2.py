import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot
from library.detect_obstacle import DetectObstacle

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 20 # total simulation duration in seconds
# Set initial state
init_state = np.array([2., 1., 0.]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

# Define Obstacles 
obst_vertices = np.array( [ [-1., 1.2], [1., 1.2], [1., 0.8], [0., 0.8], [-0.5, 0.5], \
        [-0.5, -0.5], [0.,-0.8], [1., -0.8], [1., -1.2],  [-1., -1.2], [-1., 1.2]]) 


# Robot's features: 
robot_r = 0.21 # m
max_trans_vel = 0.5 # m/s
max_rot_vel = 5 # rad/s  

# Define sensor's sensing range and resolution
sensing_range = 1. # in meter
sensor_resolution = np.pi/8 # angle between sensor data in radian

# Define parameters for switching between controllers: 
d_safe = 0.3 # m 
eps = 0.1 # m

# Controller gain
static_K = 2


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


def compute_control_input(desired_state, robot_state, current_ctrl_state, dist_reading, obst_points, x_t_s):
    # Feel free to adjust the input and output of the function as needed.
    # And make sure it is reflected inside the loop in simulate_control()

    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0., 0.]) 

    current_error = desired_state - robot_state
    u_gtg = static_K*current_error

    # Find the closest point of the wall 
    closest_point_ind = np.argmin(dist_reading)
    dist_to_closest_point = dist_reading[closest_point_ind]

    # Save closest point of the obstacle
    obstacle_coords = obst_points[:,closest_point_ind]

    #Compute avoid control
    u_avo = static_K*(robot_state[:2] - obstacle_coords)
    # Add missing variable
    u_avo = np.append(u_avo, robot_state[2])

    # Compute wall follow control in both clockwise and counterclockwise directions using rotation matrix
    u_wf_c, u_wf_cc = compute_wall_follow_rot_mat(u_avo)

    # Compute new control state, new input and "on switch position"
    current_ctrl_state, current_input, x_t_s = check_control_state(current_ctrl_state, robot_state, dist_to_closest_point, u_gtg, u_avo, u_wf_c, u_wf_cc, desired_state, x_t_s)

    current_input, current_speed = limit_velocities(current_input)

    return current_input, x_t_s, current_ctrl_state, current_speed, current_error, dist_to_closest_point

def compute_wall_follow_rot_mat(u_avo):
    # Initialize rotation matrix to clockwise direction
    rot_mat_c = np.array([[0,1],[-1,0]])

    # Clockwise wall follow
    u_wf_c = static_K*rot_mat_c.dot(u_avo[:2])
    # Counterclockwise wall follow
    u_wf_cc = -u_wf_c
    
    # Add the missing variable(s)
    u_wf_c = np.append(u_wf_c, [0])
    u_wf_cc = np.append(u_wf_cc, [0])
    
    return u_wf_c, u_wf_cc

def check_direction(u_wf_c, u_gtg):
    # Calculate the angle between u_wf_c and u_gtg
    cos_angle = (np.dot(u_gtg[:2], u_wf_c[:2]))/(np.linalg.norm(u_gtg[:2])*np.linalg.norm(u_wf_c[:2]))
    angle = np.arccos(cos_angle)

    if(angle > np.pi/2):
        return "wf_cc"
    else:
        return "wf_c"


def check_control_state(current_ctrl_state, robot_state, dist_to_closest_point, u_gtg, u_avo, u_wf_c, u_wf_cc, desired_state, x_t_s):
    if(current_ctrl_state == "gtg"):
        # Check condition 1 in lecture 9 slides (slide 14)
        if(((d_safe - eps) <= dist_to_closest_point) and ((d_safe + eps) >= dist_to_closest_point)):
            # Check whether to drive clockwise or counterclocwise
            control_state = check_direction(u_wf_c, u_gtg)
            # Save "on switch position" 
            x_t_s = robot_state[:2]

            if(control_state == "wf_c"):
                current_input = u_wf_c
                print("Switching to clockwise wall follow")
            else:
                current_input = u_wf_cc
                print("Switching to counterclockwise wall follow")
        # If condition 1 is not true --> continue with go-to-goal controller
        else: 
            control_state = current_ctrl_state
            current_input = u_gtg
    
    elif(current_ctrl_state == "avo"):
        # Check condition 1 in lecture 9 slides (slide 14)
        if(((d_safe - eps) <= dist_to_closest_point) and ((d_safe + eps) >= dist_to_closest_point)):
            # Check whether to drive clockwise or counterclocwise
            control_state = check_direction(u_wf_c, u_gtg)
            # Save "on switch position" 
            x_t_s = robot_state[:2]

            if(control_state == "wf_c"):
                current_input = u_wf_c
                print("Switching to clockwise wall follow")
            else:
                current_input = u_wf_cc
                print("Switching to counterclockwise wall follow")
        # If condition 1 is not true --> continue with avoid controller
        else: 
            control_state = current_ctrl_state
            current_input = u_avo 

    elif(current_ctrl_state == "wf_c"):
        current_distance = np.linalg.norm(robot_state[:2] - desired_state[:2])
        on_switch_distance = np.linalg.norm(x_t_s - desired_state[:2])
        # Check conditions 4 and 5 in lecture 9 slides (slide 14)
        if(np.dot(u_avo, u_gtg) > 0 and current_distance < on_switch_distance):
            control_state = "gtg"
            current_input = u_gtg
            print("Switching to go-to-goal")
        # Check condition 6 in lecture 9 slides (slide 14)
        elif(dist_to_closest_point < (d_safe - eps)):
            control_state = "avo"
            current_input = u_avo
            print("Switching to avoid")
        # If conditions 4,5 and 6 are false and the previous controller was clockwise wall follow --> continue with clockwise wall follow 
        else:
            control_state = current_ctrl_state
            current_input = u_wf_c
    
    elif(current_ctrl_state == "wf_cc"):
        current_distance = np.linalg.norm(robot_state[:2] - desired_state[:2])
        on_switch_distance = np.linalg.norm(x_t_s - desired_state[:2])
        # Check conditions 4 and 5 in lecture 9 slides (slide 14)
        if(np.dot(u_avo, u_gtg) > 0 and current_distance < on_switch_distance):
            control_state = "gtg"
            current_input = u_gtg
            print("Switching to go-to-goal")
        # Check condition 6 in lecture 9 slides (slide 14)
        elif(dist_to_closest_point < (d_safe - eps)):
            control_state = "avo"
            current_input = u_avo
            print("Switching to avoid")
        # If conditions 4,5 and 6 are false and the previous controller was clockwise wall follow --> continue with clockwise wall follow 
        else:
            control_state = current_ctrl_state
            current_input = u_wf_cc
    

    return control_state, current_input, x_t_s

def limit_velocities(current_input):
    # Calculate linear speed
    current_speed = np.linalg.norm(current_input[0:2])

    # Check that the maximum speed is not exceeded 
    if current_speed > max_trans_vel:
        current_input[0:2] = max_trans_vel * current_input[0:2]/np.linalg.norm(current_input[0:2])
        current_speed = np.linalg.norm(current_input[0:2])
    
    if current_input[2] > max_rot_vel:
        current_input[2] = max_rot_vel
    
    return current_input, current_speed


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([-2., -1., 0.]) # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 3) ) # for [vx, vy, omega] vs iteration time
    speed_history = np.zeros( (sim_iter, 1) )
    error_history = np.zeros( (sim_iter, 3) )
    min_reading_dist_history = np.zeros( (sim_iter, 1) )

    # Initiate the Obstacle Detection
    range_sensor = DetectObstacle( sensing_range, sensor_resolution)
    range_sensor.register_obstacle_bounded( obst_vertices )


    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)

        # Display the obstacle
        sim_visualizer.ax.plot( obst_vertices[:,0], obst_vertices[:,1], '--r' )
        
        # get sensor reading
        # Index 0 is in front of the robot. 
        # Index 1 is the reading for 'sensor_resolution' away (counter-clockwise) from 0, and so on for later index
        distance_reading = range_sensor.get_sensing_data( robot_state[0], robot_state[1], robot_state[2])
        # compute and plot sensor reading endpoint
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)
        pl_sens, = sim_visualizer.ax.plot(obst_points[0], obst_points[1], '.') #, marker='X')
        pl_txt = [sim_visualizer.ax.text(obst_points[0,i], obst_points[1,i], str(i)) for i in range(len(distance_reading))]

    x_t_s = np.array([0,0])
    current_ctrl_state = "gtg"
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
        current_input, x_t_s, current_ctrl_state, current_speed, current_error, minimum_reading_distance = compute_control_input(desired_state, robot_state, current_ctrl_state, distance_reading, obst_points, x_t_s)
        #------------------------------------------------------------
        
        # record the computed input at time-step t
        input_history[it] = current_input
        speed_history[it] = current_speed
        error_history[it] = current_error
        min_reading_dist_history[it] = minimum_reading_distance

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
            # update sensor visualization
            pl_sens.set_data(obst_points[0], obst_points[1])
            for i in range(len(distance_reading)): pl_txt[i].set_position((obst_points[0,i], obst_points[1,i]))
        
        #--------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts*current_input # will be used in the next iteration
        robot_state[2] = ( (robot_state[2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]
        # Update desired state if we consider moving goal position
        #desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))
    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, speed_history, error_history, min_reading_dist_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, speed_history, error_history, min_reading_dist_history = simulate_control()


    # ADDITIONAL PLOTTING
    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:,0], label='vx [m/s]')
    ax.plot(t, input_history[:,1], label='vy [m/s]')
    ax.plot(t, input_history[:,2], label='omega [rad/s]')
    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.title("Control inputs")
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
    ax.plot(t, speed_history, label='speed [m/s]')
    ax.set(xlabel="t [s]", ylabel="speed")
    plt.legend()
    plt.title("Linear speed of the robot")
    plt.grid()

    fig5 = plt.figure(5)
    ax = plt.gca()
    ax.plot(t, error_history[:,0], label='x error [m]')
    ax.plot(t, error_history[:,1], label='y error [m]')
    ax.plot(t, error_history[:,2], label='heading error [rad]')
    ax.set(xlabel="t [s]", ylabel="error")
    plt.legend()
    plt.title("State errors")
    plt.grid()

    fig6 = plt.figure(6)
    ax = plt.gca()
    ax.scatter(t, min_reading_dist_history, label='Distance [m]')
    ax.set(xlabel="t [s]", ylabel="Distance [m]")
    plt.legend()
    plt.title("Minimum reading distance from the sensor")
    plt.grid()


 



    plt.show()
