############################################################################### Author: Roozbeh Rezakhani# Email:  rrezakhani@gmail.com## Elastic material characterized by Young's modulus and Poisson's ratio###############################################################################import numpy as npclass material:        def __init__(self, E, nu, dim, twoD_type):        self.E = E        self.nu = nu        if (twoD_type == 'plane_strain'):            self.C = self.E/((1+self.nu)*(1-2*self.nu))* \                     np.array([[(1-self.nu),     self.nu,               0],                               [    self.nu, (1-self.nu),               0],                               [          0,           0, (1-2*self.nu)/2]])        elif (twoD_type == 'plane_stress'):             self.C = self.E/(1-self.nu**2)* \                     np.array([[          1,     self.nu,               0],                               [    self.nu,           1,               0],                               [          0,           0,   (1-self.nu)/2]])                           def get_C(self):        return self.C