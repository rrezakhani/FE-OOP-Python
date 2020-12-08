##############################################################################
# Author: Roozbeh Rezakhani
# Email:  rrezakhani@gmail.com
#
# This is the main file, which serves as the driver of the finite element 
# simulation. 
#
##############################################################################

import numpy as np
from numpy.linalg import inv
from gmsh_parser import gmsh_parser
from materials.material import material
from stiffness_matrix import global_stiffness_matrix

##############################################################################
# Read the input file 
input_file = open("./input.in", 'r')
line = input_file.readline()
while (line != ''):  # breaks when EOF is reached
    key = line.split(' ')[0]
    if (key == 'dim'):
        dim = int(line.split(' ')[-1])
    if (key == 'two_dimensional_problem_type'):
        two_dimensional_problem_type = line.split(' ')[-1].split('\n')[0]
    if (key == 'E'):
        E = float(line.split(' ')[-1])
    if (key == 'nu'):
        nu = float(line.split(' ')[-1])            
    if (key == 'mesh_path'):
        mesh_path = line.split(' ')[-1].split('\n')[0]
    line = input_file.readline()    

##############################################################################
# Instantiate material class and initialize material properties
mat = material(E, nu, dim, two_dimensional_problem_type)
C = mat.get_C()

##############################################################################
# Create the mesh object
# Path of the mesh file and number of problem dimensions
mesh = gmsh_parser(mesh_path, dim)
num_nodes = mesh.get_num_nodes()
elem_list = np.array(mesh.get_elems()).astype(np.int)
node_list = np.array(mesh.get_nodes()).astype(np.float)

##############################################################################
# Initialize global displacement vector
U = np.zeros(num_nodes*dim)

##############################################################################
# Apply boundary conditions
blocked = np.zeros(num_nodes*dim)
for i in range(num_nodes):
    if(node_list[i][0] == 0.0):
        blocked[2*i] = 1
        blocked[2*i+1] = 1
        
##############################################################################        
# Construct global stiffness matrix
K = np.zeros((num_nodes*dim, num_nodes*dim))
global_stiffness_matrix(K, mat, mesh)
np.savetxt("stiffness_matrix.txt", K)

##############################################################################   
# Construct the external force vector
F_ext = np.zeros(num_nodes*dim)
F_ext[1] = -20
F_ext[-1] = -20

##############################################################################   
# Iterative solve of the equilibruim equation
# Newton Raphson Method
num_load_steps = 1
itr_max = 1
itr_tol = 1E-5
res = np.zeros(num_nodes*dim)
F_int = np.zeros(num_nodes*dim)
for l in range(num_load_steps):
    F_ext = F_ext
    res = F_int - F_ext
    for k in range(itr_max):
        dU = np.dot(inv(K), -res)
        U = U + dU
        
        #=====================================================================
        # Apply DCs
        U[0]=U[1]=U[2]=U[3]=0
        
        #=====================================================================
        # Internal force vector calculation
        for e in range(len(elem_list)):         
            nodes = node_list[elem_list[e][2:]-1][:]
            neN = len(nodes)
            w_qp = np.array([1, 1, 1, 1])
            qp = np.array([[ 0.5774,  0.5774],
                           [ 0.5774, -0.5774],
                           [-0.5774,  0.5774],
                           [-0.5774, -0.5774]])
      
            Le = np.array([])
            for i in range(elem_list.shape[1]-2): 
                Le = np.concatenate((Le, [2*elem_list[e][i+2]-1, 2*elem_list[e][i+2]]))
            Le = Le.astype(np.int)
                
            u_elem = U[Le-1]
            for p in range(len(qp)):            
                xi  = qp[p][0]
                eta = qp[p][1]
                
                N = 1/4 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
                dNdxi = 1/4 * np.array([[-(1-eta), -(1-xi)],
                                        [ (1-eta), -(1+xi)],
                                        [ (1+eta),  (1+xi)],
                                        [-(1+eta),  (1-xi)]])

                J0 = np.dot(np.transpose(dNdxi), nodes)
                invJ0 = inv(J0)
                dNdx = np.dot(invJ0, np.transpose(dNdxi))       
                B = np.zeros((3, 2*neN))
                B[0, 0:2*neN+1:2] = dNdx[0,:]
                B[1, 1:2*neN+1:2] = dNdx[1,:]
                B[2, 0:2*neN+1:2] = dNdx[1,:]
                B[2, 1:2*neN+1:2] = dNdx[0,:]
            
                eps_qp = np.zeros((3,1))
                eps_qp = np.dot(B, u_elem)
                sig_qp = np.zeros((3,1))
                sig_qp = np.dot(C, eps_qp)
                Fint_qp = np.dot(np.transpose(B), sig_qp) * w_qp[p] * np.linalg.det(J0)
            
                for i in range(len(Le)):
                        F_int[Le[i]-1] = F_int[Le[i]-1] + Fint_qp[i]
        
        #=====================================================================
        res = F_int - F_ext
        tol = np.linalg.norm(res)/np.linalg.norm(F_ext)
        print("Load step {} - Iteration {} - tolerance = {}".format(i, k, tol))
        if (tol < itr_tol):
            print("Solution converged!")
            break
        if (k == itr_max-1):
            print("Maximum number of iteration is reached. Solution did NOT converge!")
            break
    
    
    
    
    
    
    
    
    
    