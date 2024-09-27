import numpy as np
import cvxopt
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 25 # total simulation duration in seconds
# Set initial state
init_state = np.array([2., 1., 0.]) # px, py, theta

# Obstacle coordinates and radius
obstacle_1 = np.array([0.85, 0.0])
obstacle_2 = np.array([-0.3, 0.1])
obstacle_3 = np.array([-1.05, -0.8])
obstacle_radius = 0.3 # m 

Rsi = 0.51 # m

static_K = 2

max_trans_vel = 0.5 # m/s
max_rot_vel = 5 # rad/s  

IS_SHOWING_2DVISUALIZATION = True

# Gamma functions used with QP 
GAMMA_FUNS = ["0.2h", "10h", "10h^3"]

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state, gamma_fun):
    
    current_error = desired_state[:2] - robot_state[:2]
    u_gtg = static_K*current_error

    current_input, h1, h2, h3 = compute_qp(robot_state, u_gtg, gamma_fun)

    current_input = limit_velocities(current_input)


    return current_input, u_gtg[:2], h1, h2, h3

def compute_qp(robot_state, u_gtg, gamma_fun):
    h1 = np.linalg.norm(robot_state[:2]-obstacle_1)**2 - Rsi**2
    h2 = np.linalg.norm(robot_state[:2]-obstacle_2)**2 - Rsi**2
    h3 = np.linalg.norm(robot_state[:2]-obstacle_3)**2 - Rsi**2

    H1 = -2*(robot_state[:2]-obstacle_1)
    H2 = -2*(robot_state[:2]-obstacle_2)
    H3 = -2*(robot_state[:2]-obstacle_3)

    if(gamma_fun == "0.2h"):
        b1 = 0.2*h1
        b2 = 0.2*h2
        b3 = 0.2*h3
    elif(gamma_fun == "10h"):
        b1 = 10.0*h1
        b2 = 10.0*h2
        b3 = 10.0*h3
    elif(gamma_fun == "10h^3"):
        b1 = 10.0*h1**3
        b2 = 10.0*h2**3
        b3 = 10.0*h3**3
    
    Q = 2*cvxopt.matrix(np.eye(2), tc = 'd')
    c = -2 * cvxopt.matrix(u_gtg[:2], tc = 'd')

    H = np.array([H1,H2,H3])
    H = cvxopt.matrix(H, tc = 'd')

    b = np.array([b1,b2,b3])
    b = cvxopt.matrix(b, tc = 'd')

    cvxopt.solvers.options["show_progress"] = False
    sol = cvxopt.solvers.qp(Q,c, H,b, verbose = False)

    u = np.array([sol['x'][0], sol['x'][1], 0])

    return u, h1, h2, h3

def limit_velocities(current_input):
    # Calculate linear speed
    current_speed = np.linalg.norm(current_input[0:2])

    # Check that the maximum speed is not exceeded 
    if current_speed > max_trans_vel:
        current_input[0:2] = max_trans_vel * current_input[0:2]/np.linalg.norm(current_input[0:2])
        current_speed = np.linalg.norm(current_input[0:2])
    
    if current_input[2] > max_rot_vel:
        current_input[2] = max_rot_vel
    
    return current_input    


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    desired_state = np.array([-2., -1., np.pi]) # numpy array for goal / the desired [px, py, theta]

    # Save states, inputs and h functions for plotting 
    state_histories = {}
    gtg_histories = {}
    input_histories = {}
    h_fun_histories = {}


    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)
        sim_visualizer.ax.add_patch(plt.Circle( obstacle_1, obstacle_radius, color="r" ))
        sim_visualizer.ax.add_patch(plt.Circle( obstacle_1, Rsi, color="r", fill = False))

        sim_visualizer.ax.add_patch(plt.Circle( obstacle_2, obstacle_radius, color="r" ))
        sim_visualizer.ax.add_patch(plt.Circle( obstacle_2, Rsi, color="r", fill = False))

        sim_visualizer.ax.add_patch(plt.Circle( obstacle_3, obstacle_radius, color="r" ))
        sim_visualizer.ax.add_patch(plt.Circle( obstacle_3, Rsi, color="r", fill = False))
    # Iterate through all gamma functions and run the simulatio
    for gamma_fun in GAMMA_FUNS:
        # Initialize robot's state (Single Integrator)
        robot_state = init_state.copy() # numpy array for [px, py, theta]
        # Store the value that needed for plotting: total step number x data length
        state_history = np.zeros( (sim_iter, len(robot_state)) ) 
        goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
        input_history = np.zeros( (sim_iter, 3) ) # for [vx, vy, omega] vs iteration time
        gtg_history = np.zeros( (sim_iter, 2) )
        h_fun_history = np.zeros( (sim_iter,3) )
        for it in range(sim_iter):
            current_time = it*Ts
            # record current state at time-step t
            state_history[it] = robot_state
            goal_history[it] = desired_state

            # COMPUTE CONTROL INPUT
            #------------------------------------------------------------
            current_input, u_gtg, h1, h2, h3 = compute_control_input(desired_state, robot_state, gamma_fun)
            #------------------------------------------------------------

            # record the computed input at time-step t
            input_history[it] = current_input
            gtg_history[it] = u_gtg
            h_fun_history[it] = [h1, h2, h3]

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

        # Save history after each gamma function 
        state_histories[gamma_fun] = state_history
        gtg_histories[gamma_fun] = gtg_history
        input_histories[gamma_fun] = input_history
        h_fun_histories[gamma_fun] = h_fun_history
    
        

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_histories, gtg_histories, input_histories, h_fun_histories


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_histories, gtg_histories, input_histories, h_fun_histories = simulate_control()


    # ADDITIONAL PLOTTING
    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]

    fig2 = plt.figure(2)
    ax = plt.gca()
    for i, gamma_fun in enumerate(input_histories):
        plt.subplot(1,3,i+1)
        plt.plot(t, input_histories[gamma_fun][:,0], label='input control vx [m/s]')
        plt.plot(t, input_histories[gamma_fun][:,1], label='input control yx [m/s]')
        plt.plot(t, gtg_histories[gamma_fun][:,0], label='go-to-goal vx [m/s]')
        plt.plot(t, gtg_histories[gamma_fun][:,1], label='go-to-goal vx [m/s]')
        plt.title(gamma_fun)
        plt.legend()
        plt.grid()
        plt.xlabel("t [s]")
        plt.ylabel("Control input")
    plt.suptitle("Control inputs vs go-to-goal controls")

    fig3 = plt.figure(3)
    for i, gamma_fun in enumerate(h_fun_histories):
        plt.subplot(1,3,i+1)
        plt.plot(t, h_fun_histories[gamma_fun][:,0], label = 'h1')
        plt.plot(t, h_fun_histories[gamma_fun][:,1], label = 'h2')
        plt.plot(t, h_fun_histories[gamma_fun][:,2], label = 'h3')
        plt.title(gamma_fun)
        plt.legend()
        plt.grid()
        plt.xlabel("t [s]")
        plt.ylabel("h function")
    plt.suptitle("h functions")

    plt.show()
