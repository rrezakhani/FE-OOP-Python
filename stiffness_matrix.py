############################################################################### Author: Roozbeh Rezakhani# Email:  rrezakhani@gmail.com## Global stiffness matrix construction###############################################################################import numpy as npfrom numpy.linalg import invfrom numpy.linalg import detdef global_stiffness_matrix(K, mat, mesh):        elem_list = np.array(mesh.get_elems()).astype(np.int)    node_list = np.array(mesh.get_nodes()).astype(np.float)    C = mat.get_C()        for e in range(len(elem_list)):                        nodes = node_list[elem_list[e][2:]-1][:]        neN = len(nodes)        # RR: method for weight and qp coordinates for different elements         # goes here        w_qp = np.array([1, 1, 1, 1])        qp = np.array([[ 0.5774,  0.5774],                       [ 0.5774, -0.5774],                       [-0.5774,  0.5774],                       [-0.5774, -0.5774]])                # Matrix used for assembly of element stiffness matrix to the         # global stiffness matrix        Le = np.array([])        for i in range(elem_list.shape[1]-2): # 2 is the first two columns about element type and phys tag            Le = np.concatenate((Le, [2*elem_list[0][i+2]-1, 2*elem_list[0][i+2]]))        Le = Le.astype(np.int)                # Loop over quadrature points to add their contribution to the global         # stiffness matrix        for p in range(len(qp)):                        xi  = qp[p][0]            eta = qp[p][1]                        # Quadrilateral element shape functions            N = 1/4 * np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])            dNdxi = 1/4 * np.array([[-(1-eta), -(1-xi)],                                    [ (1-eta), -(1+xi)],                                    [ (1+eta),  (1+xi)],                                    [-(1+eta),  (1-xi)]])            J0 = np.dot(np.transpose(dNdxi), nodes)            invJ0 = inv(J0)            dNdx = np.dot(invJ0, np.transpose(dNdxi))                        B = np.zeros((3, 2*neN))            B[0, 0:2*neN+1:2] = dNdx[0,:]            B[1, 1:2*neN+1:2] = dNdx[1,:]            B[2, 0:2*neN+1:2] = dNdx[1,:]            B[2, 1:2*neN+1:2] = dNdx[0,:]                        Kelem_qp = np.dot(np.dot(np.transpose(B), C), B) * w_qp[p] * det(J0)                        for i in range(len(Le)):                for j in range(len(Le)):                     K[Le[i]-1][Le[j]-1] = K[Le[i]-1][Le[j]-1] + Kelem_qp[i][j]                                    