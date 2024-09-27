import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 8 # total simulation duration in seconds
# Set initial state
init_state = np.array([0., 0., np.pi/2]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

k_theta = 5
k_v = 1

robot_r = 0.21 # m
wheel_r = 0.1 # m

# TODO: Change this to "a" or "b" to run the correct exercise
exercise = "b"
max_wheel_speed_rot = 10 # rad/s
l = 0.21

def limit_wheel_speed(v_control, omega):
    # Calculate wheel rotational speeds: 
    w_r = (2*v_control + omega*2*robot_r) / (2*wheel_r)
    w_l = (2*v_control - omega*2*robot_r) / (2*wheel_r)
    # Check if desired wheel speeds exceed the limit 
    if(abs(w_r) > max_wheel_speed_rot and abs(w_l) > max_wheel_speed_rot):
        # If both w_r and w_l exceeds the speed limit, calculate their current ratio and adjust both: 
        ratio = w_r/w_l
        # Set larger rotational speed to maximum and scale the other one --> ratio remains 
        if(ratio > 1): 
            w_r = w_r/abs(w_r) * max_wheel_speed_rot
            w_l = w_l/abs(w_l) * max_wheel_speed_rot
        elif(ratio < 1):
            w_l = w_l/abs(w_l) * max_wheel_speed_rot
            w_r = w_r/abs(w_r) * max_wheel_speed_rot
        else:
            w_r = w_r/abs(w_r) * max_wheel_speed_rot
            w_l = w_l/abs(w_l) * max_wheel_speed_rot
    # If only one of the wheel speeds exceeds the limits --> set the wheel speed to maximum
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

    

# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state):
    # Calculate go-to-goal linear velocity 
    v = k_v * desired_state[:2] - robot_state[:2]
    theta = robot_state[2]

    if(exercise == "a"):
        # arctan2 could work better in this one:
        theta_d = np.arctan((desired_state[1] - robot_state[1])/(desired_state[0] - robot_state[0]))
        ang_diff = theta_d - theta
        ang_diff = ( (ang_diff+ np.pi) % (2*np.pi) ) - np.pi
        # Control values: 
        omega = k_theta * (ang_diff)
        v_control = np.sqrt(v[0]**2 + v[1]**2)

    elif(exercise == "b"):
        mat1 = np.matrix([[1,0],[0, 1/l]])
        rot_mat = np.matrix([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        desired_input = mat1 @ rot_mat @ v
        v_control = desired_input[0,0]
        omega = desired_input[0,1]

    v_control, omega, w_l, w_r = limit_wheel_speed(v_control, omega)
    current_input = np.array([v_control, omega])

    return current_input, v, w_l, w_r


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([1., -1., 0.]) # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 2) ) # for [vlin, omega] vs iteration time
    speed_history = np.zeros( (sim_iter, 2) )
    wheel_rot_speed_history = np.zeros( (sim_iter, 2) )

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        # sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)

    for it in range(sim_iter):
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input, v, w_l, w_r = compute_control_input(desired_state, robot_state)
        #------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input
        speed_history[it] = v
        wheel_rot_speed_history[it] = np.array([w_l, w_r])

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
        
        #--------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of UNICYCLE model
        theta = robot_state[2]
        B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        robot_state = robot_state + Ts*(B @ current_input) # will be used in the next iteration
        robot_state[2] = ( (robot_state[2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        #desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, speed_history, wheel_rot_speed_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, speed_history, wheel_rot_speed_history = simulate_control()


    # ADDITIONAL PLOTTING
    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:,0], label='vx [m/s]')
    ax.plot(t, input_history[:,1], label='omega [rad/s]')
    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.title("Control input")
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

    # Plot historical data of state
    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, speed_history[:,0], label='u_x [m/s]')
    ax.plot(t, speed_history[:,1], label='u_y [m/s]')
    ax.set(xlabel="t [s]", ylabel="speed")
    plt.legend()
    plt.title("u_x and u_y velocities")
    plt.grid()

    fig5 = plt.figure(5)
    ax = plt.gca()
    ax.plot(t, wheel_rot_speed_history[:,0], label='w_l [rad/s]')
    ax.plot(t, wheel_rot_speed_history[:,1], label='w_r [rad/s]')
    ax.set(xlabel="t [s]", ylabel="Rotational velocities of the wheels")
    plt.legend()
    plt.title("Wheel rotational velocities")
    plt.grid()





    plt.show()
