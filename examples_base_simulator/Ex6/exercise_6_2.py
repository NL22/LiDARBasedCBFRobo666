import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 35 # total simulation duration in seconds
# Set initial state
init_state = np.array([0., 0., -np.pi/2]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-5, 5)
field_y = (-4, 4)

l = 0.1 # m
k_stat = 1 
max_wheel_speed_rot = 10 # rad/s

robot_r = 0.21 # m
wheel_r = 0.1 # m

exercise = "a"

neta = 0.3
a = 3


# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state, current_time):
    px_d, py_d, theta_d, px_d_dot, py_d_dot, px_d_dot2, py_d_dot2 = update_reference(desired_state, current_time)
    if(exercise == "a"):
        s_x = robot_state[0] + l*np.cos(robot_state[2])
        s_y = robot_state[1] + l*np.sin(robot_state[2])
        s = np.array([s_x,s_y])
        u_x_hat = k_stat*(px_d - s_x) + px_d_dot
        u_y_hat = k_stat*(py_d - s_y) + py_d_dot
        l1 = l#l*np.cos(robot_state[2])
        l2 = 0#l*np.sin(robot_state[2])
        rot_mat = np.matrix([[np.cos(robot_state[2]), -np.sin(robot_state[2])],[np.sin(robot_state[2]), np.cos(robot_state[2])]])
        l1_mat = np.matrix([[1, 0],[0, l1]])
        l2_mat = np.matrix([[1, -l2],[0, 1]])
        H = rot_mat@l1_mat@l2_mat
        u_hat = np.array([u_x_hat,u_y_hat])
        u = np.dot(np.linalg.inv(H), u_hat)
        v = u[0,0]
        omega = u[0,1]

    if(exercise == "b"):
        s = np.array([0,0])
        v_d = np.sqrt(px_d_dot**2 + py_d_dot**2)
        omega_d = (py_d_dot2*px_d_dot - px_d_dot2*py_d_dot)/(px_d_dot**2 + py_d_dot**2)
        e = compute_error_vec(robot_state, desired_state)
        k1 = 2*neta*a
        k3 = k1
        k2 = (a**2 - omega_d**2)/v_d
        u1 = -k1*e[0]
        u2 = -k2*e[1] - k3*e[2]
        v = v_d * np.cos(e[2]) - u1
        omega = omega_d - u2
    
    v_control, omega, w_l, w_r = limit_wheel_speed(v, omega)
    u = np.array([v_control, omega])
    return u, w_l, w_r, s


def update_reference(desired_state, current_time):
    px_d = desired_state[0]
    py_d = desired_state[1]
    px_d_dot = 0.5*np.sin(0.25*current_time)
    py_d_dot = -0.5*np.cos(0.5*current_time)
    theta_d = np.arctan2(px_d_dot, py_d_dot)
    px_d_dot2 = 0.25*0.5*np.cos(0.25*current_time)
    py_d_dot2 = 0.5*0.5*np.sin(0.5*current_time)

    return px_d, py_d, theta_d, px_d_dot, py_d_dot, px_d_dot2, py_d_dot2

def limit_wheel_speed(v_control, omega):
    # Calculate wheel rotational speeds: 
    w_r = (2*v_control + omega*2*robot_r) / (2*wheel_r)
    w_l = (2*v_control - omega*2*robot_r) / (2*wheel_r)
    # Check if desired wheel speeds exceed the limit 
    if(abs(w_r) > max_wheel_speed_rot and abs(w_l) > max_wheel_speed_rot):
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

def compute_error_vec(robot_state, desired_state):
    rot_mat = np.matrix([[np.cos(robot_state[2]), np.sin(robot_state[2]), 0],[-np.sin(robot_state[2]), np.cos(robot_state[2]), 0], [0, 0, 1]])
    e_hat = desired_state - robot_state
    e_hat[2] =( (e_hat[2] + np.pi) % (2*np.pi) ) - np.pi # Limit angle error between -pi and pi
    e = np.dot(rot_mat, e_hat)
    e = np.array([e[0,0], e[0,1], e[0,2]])
    return e

# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([-2*np.cos(0.25*0), -np.sin(0.5*0), np.arctan2(0.5*np.sin(0.25*0), -0.5*np.cos(0.5*0))]) # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    s_history = np.zeros( (sim_iter, 2) ) # for s vs x^d
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 2) ) # for [vlin, omega] vs iteration time
    wheel_rot_speed_history = np.zeros( (sim_iter, 2) )

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        # sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)

    for it in range(sim_iter):
        current_time = it*Ts
        desired_state = np.array([-2*np.cos(0.25*current_time), -np.sin(0.5*current_time), np.arctan2(0.5*np.sin(0.25*current_time), -0.5*np.cos(0.5*current_time))])
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input, w_l, w_r, s = compute_control_input(desired_state, robot_state, current_time)
        #------------------------------------------------------------
        # record the computed input at time-step t
        input_history[it] = current_input
        wheel_rot_speed_history[it] = np.array([w_l, w_r])
        s_history[it] = np.array([s])

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
    return state_history, goal_history, input_history, wheel_rot_speed_history, s_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history, wheel_rot_speed_history, s_history  = simulate_control()


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

    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, wheel_rot_speed_history[:,0], label='w_l [rad/s]')
    ax.plot(t, wheel_rot_speed_history[:,1], label='w_r [rad/s]')
    ax.set(xlabel="t [s]", ylabel="Rotational velocities of the wheels")
    plt.legend()
    plt.title("Wheel rotational velocities")
    plt.grid()
    print(s_history[1,1])
    if(exercise == "a"):
        fig5 = plt.figure(5)
        ax = plt.gca()
        ax.plot(t, s_history[:,0], label='s_x [m]')
        ax.plot(t, s_history[:,1], label='s_y [m]')
        ax.plot(t, goal_history[:,0], ':', label='goal px [m]')
        ax.plot(t, goal_history[:,1], ':', label='goal py [m]')
        ax.set(xlabel="t [s]", ylabel="caster state")
        plt.legend()
        plt.title("Caster states")
        plt.grid()





    plt.show()
