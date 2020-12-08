##############################################################################
# Author: Roozbeh Rezakhani
# Email:  rrezakhani@gmail.com
#
# This is the main file, which serves as the driver of the finite element 
# simulation. 
#
##############################################################################

import numpy as np
from gmsh_parser import gmsh_parser
from materials.material import material
from stiffness_matrix import global_stiffness_matrix

# Read the input file 
dim = 2
two_dimensional_problem_type = "plane_stress"

E = 3*10**7
nu = 0.3
#Gc = 2.7
#el = 0.015

# Instantiate material class
# Initialize material properties
mat = material(E, nu, dim, two_dimensional_problem_type)

# Create the mesh object
# Path of the mesh file and number of problem dimensions
mesh = gmsh_parser("./mesh/single-quad.msh", dim)
num_nodes = mesh.get_num_nodes()

# Initialize global displacement vector
U = np.zeros(num_nodes*dim)

# Apply boundary conditions
blocked = np.zeros(num_nodes*dim)
node_list = np.array(mesh.get_nodes()).astype(np.float)
for i in range(len(node_list)):
    if(node_list[i][1] == 0.0):
        blocked[2*i] = 1
        blocked[2*i+1] = 1
        
# Construct global stiffness matrix
K = np.zeros((num_nodes*dim, num_nodes*dim))
global_stiffness_matrix(K, mat, mesh)

np.savetxt("stiffness_matrix.txt", K)