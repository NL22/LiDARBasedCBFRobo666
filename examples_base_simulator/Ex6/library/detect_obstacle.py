import numpy as np

class DetectObstacle():
    def __init__(self, detect_max_dist=10, angle_res_rad=np.pi/180):
        self.__max_dist = detect_max_dist
        self.__res_rad = angle_res_rad
        self.__sens_num = round(2*np.pi/angle_res_rad)
        self.__sens_linspace = np.linspace(0., 2*np.pi, num=self.__sens_num, endpoint=False)
        # Store the obstacle as line segments (x1, y1, x2, y2)
        self.__line_segment_2D = np.zeros((0,4))
        self.__y1_min_y2, self.__x1_min_x2 = None, None

    def register_obstacle_bounded(self, vertices):
        # store list of vertices that construct the obstacle into self.__line_segment
        # expect the vertices to be numpy array N x 3 
        # TODO: assert that the last vertex should be the same as the first
        new_line_segment = np.zeros((vertices.shape[0]-1, 4))
        new_line_segment[:,:2] = vertices[:-1,:2]
        new_line_segment[:,2:] = vertices[1:,:2]
        # store the data
        self.__line_segment_2D = np.vstack((self.__line_segment_2D, new_line_segment))
        self.__update_basic_comp()

    def __update_basic_comp(self):
        self.__y1_min_y2 = self.__line_segment_2D[:,1] - self.__line_segment_2D[:,3]
        self.__x1_min_x2 = self.__line_segment_2D[:,0] - self.__line_segment_2D[:,2]

    def get_sensing_data(self, posx, posy, theta_rad):
        # The computation of detected obstacle will rely on the intersection 
        # between sensing's line-segment and obstacle's line-segment
        # The basic computation is following https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

        # Given the obstacle's line-segment denoted by (x1, y1) and (x2, y2)
        # and the sensing's line-segment denoted by (x3, y3) and (x4, y4), with
        # u = ( (x1-x3)(y1-y2) - (y1-y3)(x1-x2) ) / ( (x1-x2)(y3-y4) - (y1-y2)(x3-x4) ),
        # the obstacle is detected at (x3 + u(x4-x3), y3 + u(y4-y3))
        # if 0 <= u <= 1
        # So with (x3, y3) as the sensor's position then 
        # u is the ratio from the maximum sensing distance
        
        # number of sensing beam is m and number of obstacle line segment is n
        m = self.__sens_num
        n = self.__line_segment_2D.shape[0]
        n_0 = np.repeat(0., n)
        n_1 = np.repeat(1., n)

        sensing_angle_rad = theta_rad + self.__sens_linspace
        m_x4_min_x3 = self.__max_dist * np.cos( sensing_angle_rad )
        m_y4_min_y3 = self.__max_dist * np.sin( sensing_angle_rad )

        n_x1_min_x3 = self.__line_segment_2D[:,0] - np.repeat(posx, n)
        n_y1_min_y3 = self.__line_segment_2D[:,1] - np.repeat(posy, n)

        # Loop over each sensing direction
        u_all = np.repeat(1., m)
        for i in range(m):
            # create repmat x3 and y3 for n_obs_lseg
            n_x3_min_x4 = - np.repeat( m_x4_min_x3[i], n )
            n_y3_min_y4 = - np.repeat( m_y4_min_y3[i], n )

            t_upper = (n_x1_min_x3 * n_y3_min_y4) - (n_y1_min_y3 * n_x3_min_x4)
            u_upper = (n_x1_min_x3 * self.__y1_min_y2) - (n_y1_min_y3 * self.__x1_min_x2)
            lower = (self.__x1_min_x2 * n_y3_min_y4) - (self.__y1_min_y2 * n_x3_min_x4)
            with np.errstate(divide='ignore'):
                t = t_upper / lower
                u = u_upper / lower

            t_idx = np.logical_and( t >= n_0, t <= n_1 )
            u_idx = np.logical_and( u >= n_0, u <= n_1 )
            idx = np.logical_and( t_idx, u_idx )
            if np.any(idx): u_all[i] = min( u[idx] )

        sensing_data = self.__max_dist * u_all

        return sensing_data
