import numpy as np

# TODO: Upload to Moodle
# Define Basic Shape
triangle = np.array( [ [0., 0], [1., 1.], [0., 2.], [0., 0.] ] )
pentagon = np.array( [ [0., 0], [0.5, 0.], [1., 1.], [0.5, 2.], [0., 2.], [0., 0.] ] )
square = np.array( [ [0., 0], [1., 0.], [1., 1.], [0., 1.], [0., 0.] ] )

# Define Transformation Matrix
def rotate(theta): 
    return np.array([[ np.cos(theta), -np.sin(theta)], 
                     [ np.sin(theta), np.cos(theta)]])
def scale(kx, ky): return np.array([[ kx, 0], [ 0, ky]])
def shift(x,y): return np.array([[x, y]])

# Define Obstacles for Exercise 6.3
dict_obst_vertices = {
    'S1': (rotate(np.pi/4)@scale(1.6,1.6)@square.T).T + shift(0,-0.8*np.sqrt(2)),
    'S2': (rotate(0)@scale(1.2,1.2)@square.T).T + shift(-0.6,-0.6) + shift(2,2),
    'S3': (rotate(0)@scale(1.2,1.2)@square.T).T + shift(-0.6,-0.6) + shift(2,-2),
    'S4': (rotate(0)@scale(1.2,1.2)@square.T).T + shift(-0.6,-0.6) + shift(-2,-2),
    'S5': (rotate(0)@scale(1.2,1.2)@square.T).T + shift(-0.6,-0.6) + shift(-2,2),
}

# Another set of obstacles if you want more challenges
dict_obst_hard_mode = {
    'T1': (rotate(np.pi/6)@scale(1,0.7)@pentagon.T).T + shift(-3.5,1.8),
    'T2': (rotate(-np.pi/32)@scale(1.2,0.8)@pentagon.T).T + shift(2.5,-0),
    'T3': (rotate(np.pi/32)@scale(1.7,0.8)@pentagon.T).T + shift(-0.6,-1.2),
    'T4': pentagon + shift(3.1,-3.5),

    'S1': (scale(1,1)@square.T).T + shift(-2.8,1.9),
    'S2': (scale(1.5,1)@square.T).T + shift(-1.6,1.8),
    'S3': (scale(1.5,1.5)@square.T).T + shift(1,1),
    'S4': (scale(1,1)@square.T).T + shift(3.7,2.5),

    'S5': (scale(1,2)@square.T).T + shift(-4.8,-2),
    'S6': (rotate(-np.pi/32)@scale(0.7,1.3)@square.T).T + shift(-3.8,-1.1),
    'S7': (rotate(-np.pi/20)@scale(1,3.5)@square.T).T + shift(-2.3,-3),
    'S8': (scale(1.5,1.5)@square.T).T + shift(-0.1,-3.8),
    'S9': (scale(1.5,2.)@square.T).T + shift(1.5,-3),

    'bound': (scale(10,8)@square.T).T + shift(-5,-4),
}
