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
            if (element_gmsh_type == 2):
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
            self.F_int_elem = np.zeros(self.neN*self.dim)
            self.elas_str_ene = np.zeros((self.nqp,1))
            if (self.dim == 2):
                self.stress_total = np.zeros(self.nqp*3)
                self.strain_total = np.zeros(self.nqp*3)
                self.stress_incr  = np.zeros(self.nqp*3)
                self.strain_incr  = np.zeros(self.nqp*3)                
            else:
                self.stress_total = np.zeros(self.nqp*6)
                self.strain_total = np.zeros(self.nqp*6)
                self.stress_incr  = np.zeros(self.npp*6)
                self.strain_incr  = np.zeros(self.nqp*6)
                    
        def compute_element_stiffness(self, C):   
            w_qp = np.array([1, 1, 1, 1])
            qp = np.array([[ 0.5774,  0.5774],
                            [ 0.5774, -0.5774],
                            [-0.5774,  0.5774],
                            [-0.5774, -0.5774]])
                                
            # Loop over quadrature points to add their contribution to the global 
            # stiffness matrix
            for p in range(self.nqp):             
                xi  = qp[p][0]
                eta = qp[p][1]
                
                B, J0 = self.compute_B_J_matrices(xi, eta)
                
                K_elem_qp = np.dot(np.dot(np.transpose(B), C), B) * w_qp[p] * det(J0)
                self.K_elem = self.K_elem + K_elem_qp
            
            return self.K_elem
        
        def compute_element_internal_forces(self, C, u_elem, du_elem):
            self.F_int_elem = np.zeros(self.neN*self.dim)
            w_qp = np.array([1, 1, 1, 1])
            qp = np.array([[ 0.5774,  0.5774],
                            [ 0.5774, -0.5774],
                            [-0.5774,  0.5774],
                            [-0.5774, -0.5774]])
                                
            # Loop over quadrature points to add their contribution to the global 
            # internal forces vector
            for p in range(self.nqp):            
                xi  = qp[p][0]
                eta = qp[p][1]
                
                # compute B matrix and Jacobian of the current element at current qp
                B, J = self.compute_B_J_matrices(xi, eta)
                
                self.strain_incr[p*3:(p+1)*3]  = np.dot(B, du_elem)
                self.stress_incr[p*3:(p+1)*3]  = np.dot(C, self.strain_incr[p*3:(p+1)*3])
                self.stress_total[p*3:(p+1)*3] += self.stress_incr[p*3:(p+1)*3]
                F_int_elem_qp = np.dot(np.transpose(B), self.stress_total[p*3:(p+1)*3]) * w_qp[p] * np.linalg.det(J)
                self.F_int_elem = self.F_int_elem + F_int_elem_qp
        
            return self.F_int_elem
            
        def compute_B_J_matrices(self, xi, eta):          
            # Quadrilateral element shape functions
            # N = 1/4 * np.array([(1-xi)*(1-eta), 
            #                    (1+xi)*(1-eta), 
            #                    (1+xi)*(1+eta), 
            #                    (1-xi)*(1+eta)])
            
            dNdxi = 1/4 * np.array([[-(1-eta), -(1-xi)],
                                    [ (1-eta), -(1+xi)],
                                    [ (1+eta),  (1+xi)],
                                    [-(1+eta),  (1-xi)]])
    
            J = np.dot(np.transpose(dNdxi), self.nodes)
            invJ = inv(J)
            dNdx = np.dot(invJ, np.transpose(dNdxi))
                
            B = np.zeros((3, 2*self.neN))
            B[0, 0:2*self.neN+1:2] = dNdx[0,:]
            B[1, 1:2*self.neN+1:2] = dNdx[1,:]
            B[2, 0:2*self.neN+1:2] = dNdx[1,:]
            B[2, 1:2*self.neN+1:2] = dNdx[0,:]
            
            return B, J
        
        def compute_qp_stress(self, u_elem, du_elem):
            return u_elem    
        
        def get_elem_connectivity(self):
            return self.Le
            