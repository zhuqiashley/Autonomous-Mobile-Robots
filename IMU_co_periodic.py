import numpy
import numpy.matlib as npm
import math
import matplotlib.pyplot as plt

#Question3 Part_d
class data_area:

    def __init__(self):
        self.time = list()
        self.O_x = list()
        self.O_y = list()
        self.O_t = list()
        self.I_t = list()
        self.Co_I_t = list()
        self.G_x = list()
        self.G_y = list()
        self.Co_gps_x = list()
        self.Co_gps_y = list()

#This step is about initial state variables
class cal:

    def __init__(self, Odom_x_1, Odom_y_1, V, Odom_theta_1, Omega):

        self.delta_t = 0.001
        self.O_x = Odom_x_1
        self.O_y = Odom_y_1
        self.V = V
        self.O_t = Odom_theta_1
        self.Omega = Omega

        #state expansion
        self.A = numpy.array([ [1, 0, self.delta_t*math.cos(Odom_theta_1), 0, 0], [0, 1, self.delta_t*math.sin(Odom_theta_1), 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, self.delta_t], [0, 0, 0, 0, 1] ])

        #state uncertainty
        self.Q_Circle = numpy.array([ [.0004, 0, 0, 0, 0], [0, 0.0004, 0, 0, 0], [0, 0, .001, 0, 0], [0, 0, 0, .001, 0], [0, 0, 0, 0, .001] ])

        self.Q_Rutgers = numpy.array([ [.000000004, 0, 0, 0, 0], [0, 0.000000004, 0, 0, 0], [0, 0, .001, 0, 0], [0, 0, 0, .001, 0], [0, 0, 0, 0, .001] ])

        #transition matrix
        self.H = numpy.array([ [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1] ])

        #measurement uncertainty
        self.R = numpy.array([ [.04, 0, 0, 0, 0], [0, .04, 0, 0, 0], [0, 0, .01, 0, 0], [0, 0, 0, .01, 0], [0, 0, 0, 0, .01] ])


        self.B = numpy.array([ [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1] ])

        self.u = numpy.array( [0, 0, 0, 0, 0] )

        self.P = numpy.array([ [.001, 0, 0, 0, 0], [0, .001, 0, 0, 0], [0, 0, .001, 0, 0], [0, 0, 0, .001, 0], [0, 0, 0, 0, .001] ])

#reads values from file
def read_file( field ):

    file1 = open('EKF_DATA_circle.txt', 'r')

    count = 0
    for line in lines:
        if count != 0:
            data = line.split(',')
            #print(data)
            field.time.append( float(data[0]) )
            field.O_x.append( float(data[1]) )
            field.O_y.append( float(data[2]) )
            field.O_t.append( float(data[3]) )
            field.I_t.append( float(data[4]) )
            field.Co_I_t.append( float(data[5]) )
            field.G_x.append( float(data[6]) )
            field.G_y.append( float(data[7]) )
            field.Co_gps_x.append( float(data[8]) )
            field.Co_gps_y.append( float(data[9]) )
        count += 1

if __name__=="__main__":

    file1 = open('EKF_DATA_circle.txt', 'r')

    lines = file1.readlines()

    field = data_area()

    read_file( field )

    noise_mean = 0.5;
    noise_std = 0.1;

    total = len(field.time)


    #Gps position continuous Gaussian noise
    Gps_x = field.G_x
    Gps_y = field.G_y

    #Gps covariance continuous Gaussian noise
    Gps_Co_x = field.Co_gps_x
    Gps_Co_y = field.Co_gps_y

    #odometry position continuous Gaussian noise
    O_x = field.O_x
    O_y = field.G_y

    #Add noise to IMU covariance
    IMU = [ [numpy.random.normal() for i in range(2)] for j in range(total) ]
    IMU = numpy.array(IMU)
    IMU_noise = (noise_std * IMU) + (noise_mean * numpy.ones((total,2), dtype=float) )
    IMU_Co = IMU_noise[:,1]
    IMU_Co_heading = list()
    #IMU periodic noise
    for i in range(total):
        if (i >= 500 and i <= 1000) or (i >= 1500 and i <= 2000) or (i >= 2500 and i <= 3000):
            IMU_Co_heading.append(field.Co_I_t[i] + IMU_Co[i])
        else:
            IMU_Co_heading.append(field.Co_I_t[i])

    #Calibrate IMU to match with the robot's heading initially
    IMU_heading = field.I_t+(0.32981-0.637156)*numpy.ones((total), dtype=float)

    V = 0.14

    #Distance between 2 wheels
    L = 1#meters

    #Angular Velocity
    Omega = V*math.tan(field.O_t[0])/L

    #set time_step
    delta_t = 0.001

    s = cal( O_x[0], O_y[0], V, field.O_t[0], Omega )

    Z_k = numpy.array( [Gps_x[0], Gps_y[0], V, IMU_heading[0], Omega] )

    x_k = numpy.array( [Gps_x[0], Gps_y[0], V, IMU_heading[0], Omega] )

    P_k = s.P
    t = 1

    plt.scatter( 0, field.O_t[0], s=0.4, c='red', label='Odometry_heading' )
    plt.scatter( 0, IMU_heading[0], s=0.4, c='green', label='IMU_heading' )
    plt.scatter( 0, x_k[3], s=0.4, c='blue', label='Kalman_Filter' )

    #Kalman filter implementation
    for t in range(total):

        #create kalman filter:
        priori_x_k = numpy.add( s.A.dot(x_k), s.B.dot( s.u ) ) #+ s.Q_Circle	#priori state(-X(k))
        priori_P_k = numpy.add( s.A.dot(P_k.dot(numpy.transpose(s.A))), s.Q_Circle )

        Y_k = s.H.dot(priori_x_k) #+ s.R	#Observation Prediction(Y(k))
        kalman_gain_inv = numpy.linalg.inv( numpy.add(s.H.dot(priori_P_k.dot(numpy.transpose(s.H))), s.R) )
        K_k = priori_P_k.dot(numpy.transpose(s.H).dot(kalman_gain_inv)) #kalman gain

        x_k = numpy.add( priori_x_k, K_k.dot( numpy.subtract(Z_k, Y_k) ) )	#Final Output
        #print( x_k )

        plt.scatter( t, field.O_t[t], s=0.4, c='red' )
        plt.scatter( t, IMU_heading[t], s=0.4, c='green' )
        plt.scatter( t, x_k[3], s=0.4, c='blue' )

        #update values:
        P_k = numpy.subtract( priori_P_k, K_k.dot( s.H.dot( priori_P_k ) ) ) #updated P_k
        s.A = numpy.array([ [1, 0, delta_t*math.cos(field.O_t[t]), 0, 0], [0, 1, delta_t*math.sin(field.O_t[t]), 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, delta_t], [0, 0, 0, 0, 1] ])
        s.R = numpy.array([ [Gps_Co_x[t], 0, 0, 0, 0], [0, Gps_Co_y[t], 0, 0, 0], [0, 0, .01, 0, 0], [0, 0, 0, IMU_Co_heading[t], 0], [0, 0, 0, 0, .01] ])
        Z_k = numpy.transpose( [Gps_x[t], Gps_y[t], V, IMU_heading[t], Omega] )

    plt.legend()
    plt.show()
