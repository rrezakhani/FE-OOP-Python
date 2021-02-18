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

from src.solid_mechanics_model import solid_mechanics_model
from src.element_PF import element_PF
from src.gmsh_parser import gmsh_parser
from src.materials.material_PF import material_PF
from src.vtk_writer import vtk_writer

##############################################################################
# Read the input file 
input_file = open("./input-PhaseField.in", 'r')
line = input_file.readline()
disp_BC = []
trac_BC = []
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
    if (key == 'Gc'):
        Gc = float(line.split(' ')[-1])
    if (key == 'el'):
        el = float(line.split(' ')[-1])         
    if (key == 'mesh_path'):
        mesh_path = line.split(' ')[-1].split('\n')[0]
    if (key == 'Dirichlet_BC'):
        disp_BC.append([line.split(' ')[-3][1:], 
                        line.split(' ')[-2], 
                        line.split(' ')[-1].split('\n')[0][1:-2]])
    if (key == 'analysis_type'):
        analysis_type = line.split(' ')[-1].split('\n')[0]
    if (key == 'solver'):
        solver = line.split(' ')[-1].split('\n')[0]
    line = input_file.readline()    

##############################################################################
# Create the mesh object
# Path of the mesh file and number of problem dimensions
mesh = gmsh_parser(mesh_path, dim)
num_nodes = mesh.get_num_nodes()
elem_list = np.array(mesh.get_elems()).astype(np.int)
node_list = np.array(mesh.get_nodes()).astype(np.float)

elem_obj_list = []
for e in range(len(elem_list)):
    
    nodes = node_list[elem_list[e][2:]-1][:,:dim]
    neN = len(nodes)
    element_gmsh_type = elem_list[e,0]
    
    # Element connectivity
    Le = np.array([])
    LePF = np.array([])
    for i in range(elem_list.shape[1]-2): 
        Le = np.concatenate((Le, [2*elem_list[e][i+2]-1, 2*elem_list[e][i+2]]))
        LePF = np.concatenate((LePF, [elem_list[e][i+2]]))
    Le = Le.astype(np.int)
    LePF = LePF.astype(np.int)
    
    elem_obj = element_PF(element_gmsh_type, neN, dim, nodes, Le, LePF)
    elem_obj_list.append(elem_obj)

##############################################################################
# Instantiate material class and initialize material properties
mat = material_PF(E, nu, Gc, el, dim, two_dimensional_problem_type)
C = mat.get_C()

##############################################################################
# Instantiate model class to build the general framework
solid_mechanics_model = solid_mechanics_model(mat, mesh, dim)

##############################################################################
# Initialize global displacement vector
U = np.zeros(num_nodes*dim)
dU = np.zeros(num_nodes*dim)

##############################################################################        
# Construct global stiffness matrix
#K = solid_mechanics_model.construct_stiffness_matrix()

##############################################################################
# Boundary condition arrays
blocked = np.zeros(num_nodes*dim)
u_bar = np.zeros(num_nodes*dim)

# Apply prescribed boundary conditions
phys_array = mesh.get_phys_array()
bndry_elems = np.array(mesh.get_bndry_elems()).astype(np.int)   
 
for d_BC in disp_BC:
    val = float(d_BC[0])
    comp = int(d_BC[1])
    phys_tag = d_BC[2]
    for i in range(len(phys_array)):
        if(phys_array[i][2] == phys_tag):
            phys_index = int(phys_array[i][1])
    phys_elems = bndry_elems[bndry_elems[:,1]==phys_index][:,2:]
    blocked[2*(np.unique(phys_elems)-1)+comp-1] = 1
    u_bar[2*(np.unique(phys_elems)-1)+comp-1] = val

##############################################################################   
# Construct the external force vector
F_ext = np.zeros(num_nodes*dim)

# Construct the reaction force vector
R_ext = np.zeros(num_nodes*dim)

##############################################################################   
# Construct PF (phase field) related variables
phi = np.zeros(num_nodes)
K_phi = np.zeros((num_nodes, num_nodes))
res_phi = np.zeros(num_nodes)

##############################################################################   
# Iterative solve of the equilibruim equation
# Newton Raphson Method
num_load_steps = 50
itr_max = 10
itr_tol = 1E-5
F_int = np.zeros(num_nodes*dim)
res = np.zeros(num_nodes*dim)
F_ext_ = np.zeros(num_nodes*dim)
u_bar_ = np.zeros(num_nodes*dim)

vtk_writer("./results", '0', mesh, U, phi, F_int)

for l in range(num_load_steps):
    
    #=====================================================================
    # update essential and neumann boundary conditions
    u_bar_ += 1/num_load_steps * u_bar
    F_ext_ += 1/num_load_steps * F_ext
    
    #print(u_bar_)
    #phi = np.zeros(num_nodes)
        
    # compute internal forces vector
    #F_int = solid_mechanics_model.compute_PF_internal_forces(U, dU, phi)
    
    #=====================================================================
    # update stiffness matrix of the displacement governing equation
    K = solid_mechanics_model.update_PF_stiffness_matrix(phi)
    #print(K)
    
    #=====================================================================
    # compute residual of the displacement governing equation
    res = F_ext_ + R_ext - F_int
    
    #=====================================================================
    # Impose essential boundary conditions
    K_temp = np.copy(K)
    for m in range(num_nodes*dim):
        if(blocked[m]==1):
            for n in range(num_nodes*dim):
                if(blocked[n]==0):
                    res[n] -= K[n,m] * u_bar_[m]
            K_temp[m,:] = 0.0
            K_temp[:,m] = 0.0
            K_temp[m,m] = 1.0
            res[m] = u_bar_[m]
    
    #=====================================================================
    # Loop on iterations for the DISPLACEMENT governing equation
    max_itr_reached = False
    for k in range(itr_max):
        
        # calculate displacement vector increment
        U = np.dot(inv(K_temp), res)    
        
        #print(U)
        
        # update the reaction forces
        # R_ext = np.zeros(num_nodes*dim)
        for m in range(num_nodes*dim):
            if(blocked[m]==1):
                R_ext[m] = np.dot(K[m,:], U)       
        #print(R_ext)
        
        # update the displacement field
        #U = U + dU
               
        # compute internal forces vector
        F_int = solid_mechanics_model.compute_PF_internal_forces(U, dU, phi)
        #print(F_int)
        
        # update residual and check convergence
        res = F_ext_ + R_ext - F_int        
        tol = np.linalg.norm(res)/np.linalg.norm(F_ext_ + R_ext)
        
        # printing iteration information
        print("Load step {} - Iteration {} - tolerance = {}".format(l+1, k+1, tol))
        if (tol < itr_tol):
            print("Displacement solve converged!")
            break # break out of the iteration loop to the next load step
        if (k == itr_max-1):
            print("Maximum number of iteration in displacement solve is reached!")
            print("Displacement solve did NOT converge!")
            max_itr_reached = True
            break # break out of the iteration loop
    if(max_itr_reached):
        break # break out of the load step loop
      
    #=====================================================================
    # compute residual of the PF (phase field) governing equation
    res_phi = solid_mechanics_model.compute_PF_residual(phi)

    #=====================================================================
    # update stiffness matrix of the PF (phase field) governing equation
    K_phi = solid_mechanics_model.compute_PF_stiffness_matrix()  
    
    #=====================================================================
    # Loop on iterations for the PF (phase field) governing equation
    for k in range(itr_max):
        
        # calculate phase field vector increment
        dphi = np.dot(inv(K_phi), -res_phi)        
        
        # update the phase field vector
        phi = phi + dphi
               
        # update residual and check convergence
        res_phi= solid_mechanics_model.compute_PF_residual(phi)        
        tol = np.linalg.norm(res_phi)
        
        # printing iteration information
        print("Phase field solve step {} - Iteration {} - tolerance = {}".format(l+1, k+1, tol))
        if (tol < itr_tol):
            print("Phase field solve converged!")
            break # break out of the iteration loop to the next load step
        if (k == itr_max-1):
            print("Maximum number of iteration in displacement solve is reached!")
            print("Phase field solve did NOT converge!")
            max_itr_reached = True
            break # break out of the iteration loop
    if(max_itr_reached):
        break # break out of the load step loop
    
    #=====================================================================
    # write vtk file   
    vtk_writer("./results", l+1, mesh, U, phi, F_int)
 
    
    
    
    
    
    
    