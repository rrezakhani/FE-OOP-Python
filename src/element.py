##############################################################################
# Author: Roozbeh Rezakhani
# Email:  rrezakhani@gmail.com
#
# This is the Element class.
#
##############################################################################

import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

class element:
    
        def __init__(self, element_gmsh_type, neN, dim, nodes, Le):
            if (element_gmsh_type == 1):
                self.elem_type = '2node-line'
            if (element_gmsh_type == 3):
                self.elem_type = '3node-triangular'
                self.nqp = 1
            if (element_gmsh_type == 3):
                self.elem_type = '4node-quadrilateral'
                self.nqp = 4
            self.neN = neN
            self.dim = dim
            self.nodes = nodes
            self.Le = Le
            self.K_elem = np.zeros((self.neN*self.dim, self.neN*self.dim))
            self.elas_str_ene = np.zeros((self.nqp,1))
            if (self.dim == 2):
                self.stress_total = np.zeros((self.nqp*3,1))
                self.strain_total = np.zeros((self.nqp*3,1))
                self.stress_incr  = np.zeros((self.nqp*3,1))
                self.strain_incr  = np.zeros((self.nqp*3,1))                
            else:
                self.stress_total = np.zeros((self.nqp*6,1))
                self.strain_total = np.zeros((self.nqp*6,1))
                self.stress_incr  = np.zeros((self.npp*6,1))
                self.strain_incr  = np.zeros((self.nqp*6,1))
            
        def compute_B_J_matrices(self, xi, eta):
            
            # Quadrilateral element shape functions
            N = 1/4 * np.array([(1-xi)*(1-eta), 
                                (1+xi)*(1-eta), 
                                (1+xi)*(1+eta), 
                                (1-xi)*(1+eta)])
            
            dNdxi = 1/4 * np.array([[-(1-eta), -(1-xi)],
                                    [ (1-eta), -(1+xi)],
                                    [ (1+eta),  (1+xi)],
                                    [-(1+eta),  (1-xi)]])
    
            J0 = np.dot(np.transpose(dNdxi), self.nodes)
            invJ0 = inv(J0)
            dNdx = np.dot(invJ0, np.transpose(dNdxi))
                
            B = np.zeros((3, 2*self.neN))
            B[0, 0:2*self.neN+1:2] = dNdx[0,:]
            B[1, 1:2*self.neN+1:2] = dNdx[1,:]
            B[2, 0:2*self.neN+1:2] = dNdx[1,:]
            B[2, 1:2*self.neN+1:2] = dNdx[0,:]
            
            return B, J0
        
        def compute_element_stiffness(self, C):
            
            w_qp = np.array([1, 1, 1, 1])
            qp = np.array([[ 0.5774,  0.5774],
                            [ 0.5774, -0.5774],
                            [-0.5774,  0.5774],
                            [-0.5774, -0.5774]])
                                
            # Loop over quadrature points to add their contribution to the global 
            # stiffness matrix
            for p in range(len(qp)):
                
                xi  = qp[p][0]
                eta = qp[p][1]
                
                B, J0 = self.compute_B_J_matrices(xi, eta)
                
                K_elem_qp = np.dot(np.dot(np.transpose(B), C), B) * w_qp[p] * det(J0)
                self.K_elem = self.K_elem + K_elem_qp
            
            return self.K_elem