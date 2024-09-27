import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 15 # total simulation duration in seconds
# Set initial state
init_state = np.array([1.5, -1, np.pi/2]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

robot_r = 0.21 # m
max_trans_vel = 0.5 # m/s
max_rot_vel = 5 # rad/s

d_safe = 0.8 # m 
eps = 0.3 # m

static_K = 2 # Gain for controllers

obstacle_coords = np.array([0, 0, 0])

# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state, current_ctrl_state):
    # Feel free to adjust the input and output of the function as needed.
    # And make sure it is reflected inside the loop in simulate_control()

    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0., 0.]) 
    
    # Calculate the current distance between the robot and the obstacle
    dist_to_obstacle = np.linalg.norm(robot_state - obstacle_coords)

    if current_ctrl_state == "gtg":
        # Check if the robot is too close to the obstacle --> if it is, change to avo controller otherwise keep computing gtg control
        if dist_to_obstacle < d_safe:
            current_input, speed = compute_avo(robot_state)
            current_ctrl_state = "avo"
        else: 
            current_input, speed = compute_gtg(desired_state, robot_state)

    elif current_ctrl_state == "avo":
        # Check if the robot is far enough from the obstacle --> if it is, change to gtg controller otherwise keep computing avo control
        if dist_to_obstacle > d_safe + eps:
            current_input, speed = compute_gtg(desired_state, robot_state)
            current_ctrl_state = "gtg"
        else:
            current_input, speed = compute_avo(robot_state)
    
    # When the simulation starts, first check if it's safe for the robot to start gtg control 
    else:
        if dist_to_obstacle >= d_safe:
            current_input, speed = compute_gtg(desired_state, robot_state)
            current_ctrl_state = "gtg"
        else:
            current_input, speed = compute_avo(robot_state)
            current_ctrl_state = "avo"
    
    error = robot_state - desired_state

    
    return current_input, current_ctrl_state, speed, dist_to_obstacle, error


def compute_avo(robot_state):
    u_avo = static_K*(robot_state - obstacle_coords)
    speed = np.linalg.norm(u_avo[0:2])
    # Check that the maximum speed is not exceeded 
    if speed > max_trans_vel:
        u_avo[0:2] = max_trans_vel * u_avo[0:2]/np.linalg.norm(u_avo[0:2])
        speed = np.linalg.norm(u_avo[0:2])

    if u_avo[2] > max_rot_vel:
        u_avo[2] = max_rot_vel
    return u_avo, speed



def compute_gtg(desired_state, robot_state):
    u_gtg = static_K*(desired_state - robot_state)
    speed = np.linalg.norm(u_gtg[0:2])

    # Check that the maximum speed is not exceeded 
    if speed > max_trans_vel:
        u_gtg[0:2] = max_trans_vel * u_gtg[0:2]/np.linalg.norm(u_gtg[0:2])
        speed = np.linalg.norm(u_gtg[0:2])
    
    if u_gtg[2] > max_rot_vel:
        u_gtg[2] = max_rot_vel
    return u_gtg, speed

# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([-2, 0.5, 0]) # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 3) ) # for [vx, vy, omega] vs iteration time
    speed_history = np.zeros( (sim_iter,1) )
    dist_to_obstacle_history = np.zeros( (sim_iter,1 ) )
    error_history = np.zeros( (sim_iter, 3) )

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)
        sim_visualizer.ax.add_patch(plt.Circle( (0,0), 0.5, color="r" ))
        sim_visualizer.ax.add_patch(plt.Circle( (0,0), d_safe, color="r", fill = False))
        sim_visualizer.ax.add_patch(plt.Circle( (0,0), d_safe+eps, color="g", fill = False))

    # In the beginning, we need to check if it's safe to start gtg controller 
    current_ctrl_state = ""
    for it in range(sim_iter):
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state



        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input, current_ctrl_state, speed, dist_to_obstacle, error = compute_control_input(desired_state, robot_state, current_ctrl_state)
        #------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input
        speed_history[it] = speed
        dist_to_obstacle_history[it] = dist_to_obstacle
        error_history[it] = error

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
        
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
    return state_history, goal_history, input_history, speed_history, dist_to_obstacle_history, error_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, speed_history, dist_to_obstacle_history, error_history = simulate_control()


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
    plt.title("Robot states")
    plt.grid()

    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, speed_history, label='robot speed')
    ax.set(xlabel="t [s]", ylabel="Speed")
    # TODO: Add max speed 
    plt.title("Robot speeds")
    plt.grid()

    fig5 = plt.figure(5)
    ax = plt.gca()
    ax.plot(t, dist_to_obstacle_history)
    # TODO: Add minimum allowed distance (?)
    ax.set(xlabel="t [s]", ylabel="Distance")
    plt.title("Distance to obstacle")
    plt.grid()

    fig6 = plt.figure(6)
    ax = plt.gca()
    ax.plot(t, error_history[:,0], label = 'x error [m]')
    ax.plot(t, error_history[:,1], label = 'y error [m]')
    ax.plot(t, error_history[:,2], label = 'heading error [rad]')
    ax.set(xlabel="t [s]", ylabel="error")
    plt.legend()
    plt.title("State error")
    plt.grid()

    plt.show()
