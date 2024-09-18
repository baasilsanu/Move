#Solves the SSD set of equations

#eqn : du/dt = R - r_m*U
#eqn : R = 0.25 * k * (k_plus_square - k_e_square) * C_13 
#eqn : dC/dt = A(U)*C + C*Transpose(A(U)) + eps*Q
#eqn : A(U) = W + UL

#eqn : <WC> + <CW^T> + <ULC> + <UCL^T> + epsilon * Q (Q = <eta * eta ^ T>) 

import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time):

        #Make sure this is updated

        self.epsilon = epsilon
        self.N_0_squared = N_0_squared
        self.r_m = r_m
        self.k = k
        self.m = m
        self.m_u = m_u
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
        self.k_e_square = k**2 + m**2
        self.k_plus_square = k**2 + (m + m_u)**2

        self.W_e = np.array([[-1, (k / self.k_e_square)], [-k * N_0_squared, -1]])
        self.W_plus = np.array([[-1, -k / self.k_plus_square], [k * N_0_squared, -1]])
        self.L_e_plus = np.array([[(-k / (2 * self.k_e_square)) * (self.k_plus_square - m_u**2), 0],
                                  [0, k / 2]])
        self.L_plus_e = np.array([[(-k / (2 * self.k_plus_square)) * (m_u**2 - self.k_e_square), 0],
                                  [0, -k / 2]]) 
        self.U = 0.55053473999083
        # self.U = 0.01
        # self.C = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.C = np.array([[0.0001490152423865438,
-0.056342960706027546,
4.705459608940639e-06,
-0.023997387342599555],
[-0.056342960706027546,
153.61445123375822,
-0.043334023121050586,
-12.060326483759946],
[4.705459608940639e-06,
-0.043334023121050586,
4.7701541745783226e-05,
-0.01911624053635777],
[-0.023997387342599555,
-12.060326483759946,
-0.01911624053635777,
56.7986445738767]])


        self.A_U = np.array([[-1, (k / self.k_e_square), self.U*(-k / (2 * self.k_e_square)) * (self.k_plus_square - m_u**2), self.U*0],
                             
                             [-k * N_0_squared, -1, self.U*0, self.U*self.k/2],

                             [self.U*(-k / (2 * self.k_plus_square)) * (m_u**2 - self.k_e_square), self.U*0, -1, -self.k/self.k_plus_square],

                             [self.U*0, self.U* -k / 2,k * N_0_squared, -1]])  
    
        self.Q = np.array([[8/self.k_e_square, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.R = 0
        self.C_History = np.zeros((self.num_steps, 4, 4))              
        # self.C_History = np.zeros(self.num_steps)
        self.U_History = np.zeros(self.num_steps)
        self.R_History = np.zeros(self.num_steps)
        self.C = self.C.astype(np.float64)



        self.p = self.k / self.k_e_square
        self.q = -self.k / self.k_plus_square
        self.r = self.k * self.N_0_squared
        self.s = -self.k * (self.k_plus_square - self.m_u**2) / (2*k_e_square)
        self.t = -self.k * (m_u**2 - self.k_e_square) / (2 * self.k_plus_square)
        self.v = self.k / 2

        self.bigW = np.array([[-1, self.p, 0, 0], [-self.r, -1, 0, 0], [0, 0, -1, self.q], [0, 0, self.r, -1]])
        self.bigL = np.array([[0, 0, self.s, 0], [0, 0, 0, self.v], [self.t, 0, 0, 0], [0, -self.v, 0, 0]])




        self.U_3Corr = 0.55053473999083
        # self.U_3Corr = 0.01
        self.R_3Corr = 0
        self.C_3Corr = np.array([[0.0001490152423865438,
-0.056342960706027546,
4.705459608940639e-06,
-0.023997387342599555],
[-0.056342960706027546,
153.61445123375822,
-0.043334023121050586,
-12.060326483759946],
[4.705459608940639e-06,
-0.043334023121050586,
4.7701541745783226e-05,
-0.01911624053635777],
[-0.023997387342599555,
-12.060326483759946,
-0.01911624053635777,
56.7986445738767]])
        # self.C_3Corr = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.C_3Corr = self.C_3Corr.astype(np.float64)

        self.C_History_3Corr = np.zeros((self.num_steps, 4, 4))
        self.U_History_3Corr = np.zeros(self.num_steps)
        self.R_History_3Corr = np.zeros(self.num_steps)
        

    def simulate(self):
        for i in range(self.num_steps):
            # randArr = np.array([(2 * np.sqrt(2) * np.random.normal(0, 1))/np.sqrt(self.k_e_square), 0, 0, 0])
            bigQ = np.array([[8/k_e_square, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]])

            self.A_U = np.array([[-1, (k / self.k_e_square), self.U*(-k / (2 * self.k_e_square)) * (self.k_plus_square - m_u**2), self.U*0],
                             
                             [-k * N_0_squared, -1, self.U*0, self.U*self.k/2],

                             [self.U*(-k / (2 * self.k_plus_square)) * (m_u**2 - self.k_e_square), self.U*0, -1, -self.k/self.k_plus_square],

                             [self.U*0, self.U* -k / 2,k * N_0_squared, -1]]) 
                             
 
            C_dot = (self.A_U @ self.C + self.C @ np.transpose(self.A_U) + self.epsilon * self.Q) 
            U_dot = (self.R - self.r_m * self.U)

            C_3Corr_dot = self.bigW @ self.C_3Corr + self.C_3Corr @ np.transpose(self.bigW) + self.U_3Corr * (self.bigL @ self.C_3Corr) + self.U_3Corr * (self.C_3Corr @ np.transpose(self.bigL)) + (self.epsilon * bigQ)  #fix the noise
            U_3Corr_dot = self.R_3Corr - self.r_m * self.U_3Corr



            self.C_History[i] = self.C
            self.U_History[i] = self.U
            self.U_History_3Corr[i] = self.U_3Corr
            self.R_History[i] = self.R

            self.R_History_3Corr[i] = self.R_3Corr

    def make_plots(self):

        time_array = np.arange(0, self.total_time, .001)
        fig, axs = plt.subplots(1, 2, figsize = (15, 5))

       
        axs[0].plot(time_array, self.U_History)
        axs[0].plot(time_array, self.U_History_3Corr)
        axs[0].set_title(f"U History")
        axs[0].grid()
        axs[1].plot(time_array, self.R_History)
        axs[1].plot(time_array, self.R_History_3Corr)
        axs[1].set_title(f"R Values")
        axs[1].grid()


        plt.show()


if __name__ == "__main__":
    epsilon = 0.12394270273516043
    N_0_squared = 318.8640217310387
    # epsilon = 0.01
    # N_0_squared = 100
    r_m = 0.1
    k = 2 * np.pi * 6
    m = 2 * np.pi * 3
    m_u = 2 * np.pi * 7
    dt = 0.001
    total_time = 200

    k_e_square = k**2 + m**2

    C_11 = (2*epsilon/k_e_square)*(2-(np.square(k) * N_0_squared/(k_e_square + np.square(k)*N_0_squared)))
    C_12 = -(2*epsilon*k*N_0_squared)/(k_e_square + np.square(k)*N_0_squared)
    C_22 = (2*epsilon*np.square(k)*np.square(N_0_squared))/(k_e_square + np.square(k)*N_0_squared)


    sim = Simulation(epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time)
    sim.simulate() 
    print(C_11, C_12, C_22)
    print(sim.U_History[-1])
    print(sim.C_History[-1])
    sim.make_plots()
    

