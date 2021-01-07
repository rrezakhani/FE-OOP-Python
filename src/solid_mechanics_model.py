##############################################################################
# Author: Roozbeh Rezakhani
# Email:  rrezakhani@gmail.com
#
# This is the solid_mechanics_model class.
#
##############################################################################

import numpy as np
from src.element import element 

class solid_mechanics_model:
    
    def __init__(self, mat, mesh):
        
        print('Solid Mechanics Model object is constructed!')
        self.mat = mat
        self.mesh = mesh
        
    def construct_stiffness_matrix(self):
        
        elem_list = np.array(self.mesh.get_elems()).astype(np.int)
        node_list = np.array(self.mesh.get_nodes()).astype(np.float)
        num_nodes = self.mesh.get_num_nodes()
        dim = self.mesh.get_dim()
        C = self.mat.get_C()
        K = np.zeros((num_nodes*dim, num_nodes*dim))
        
        for e in range(len(elem_list)):
                        
            nodes = node_list[elem_list[e][2:]-1][:,:dim]
            neN = len(nodes)
            element_gmsh_type = elem_list[e,0]
            
            # Element connectivity
            Le = np.array([])
            for i in range(elem_list.shape[1]-2): # 2 is the first two columns about element type and phys tag
                Le = np.concatenate((Le, [2*elem_list[e][i+2]-1, 2*elem_list[e][i+2]]))
            Le = Le.astype(np.int)
            
            elem = element(element_gmsh_type, neN, dim, nodes, Le)
            
            # Element stiffness matrix        
            K_elem = elem.compute_element_stiffness(C)
                
            for i in range(len(Le)):
                for j in range(len(Le)): 
                    K[Le[i]-1][Le[j]-1] = K[Le[i]-1][Le[j]-1] + K_elem[i][j]
        
        print('Stiffness matrix is constructed!')
        return K

