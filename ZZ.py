#The Zienkiewicz-Zhu Superconvergent Patch Recovery Method for P1-P1 Elements (not set up for higher order)
import sys
import os
from pathlib import Path
import vtk
from vtkmodules.vtkFiltersParallelDIY2 import vtkRedistributeDataSetFilter
import numpy as np
import pyvista as pv
import h5py
import math
import multiprocess as mp
import time
import scipy
import pandas as pd
from numba import njit

nprocs = 80
nprocs1 = 79

class Dataset():
    """ Load BSL-specific data and common ops. 
    """
    def __init__(self, folder, file_glob_key=None):
        self.folder = Path(folder)

        file_glob_key = '*_curcyc_*up.h5'
        mesh_glob_key = '*h5'

        self.up_files = sorted(folder.glob(file_glob_key), 
            key=self._get_ts)
        
        self.mesh_file = sorted(folder.glob(mesh_glob_key), 
            key=lambda x: len(x.stem))[0]

    def __call__(self, u_file, array='u'):
        """ Return velocity in u_file. """
        if array in ['u', 'p']:
            h5_file = u_file
            with h5py.File(h5_file, 'r') as hf:
                val = np.array(hf['Solution'][array])
        
        return val

    def _get_ts(self, h5_file):
        """ Given a simulation h5_file, get ts. """
        return int(h5_file.stem.split('_ts=')[1].split('_')[0])
        
    def _get_time(self, h5_file):
        """ Given a simulation h5_file, get time. """
        return float(h5_file.stem.split('_t=')[1].split('_ts')[0]) / 1000.0
        
    def assemble_mesh(self):
        """ Create UnstructuredGrid from h5 mesh file. """
        assert self.mesh_file.exists(), 'mesh_file does not exist.'
        
        with h5py.File(self.mesh_file, 'r') as hf:
            points = np.array(hf['Mesh']['coordinates'])*(10**-3)
            cells = np.array(hf['Mesh']['topology'])

            celltypes = np.empty(cells.shape[0], dtype=np.uint8)
            celltypes[:] = vtk.VTK_TETRA
            cell_type = np.ones((cells.shape[0], 1), dtype=int) * 4
            cells = np.concatenate([cell_type, cells], axis = 1)
            self.mesh = pv.UnstructuredGrid(cells.ravel(), celltypes, points)
            self.surf = self.mesh.extract_surface()
    
        # self.assemble_surface()
        return self
    
    def assemble_surface(self, mesh_file=None):
        """ Create PolyData from h5 mesh file. 

        Args:
            mesh_file
        """
        if mesh_file is None:
            mesh_file = self.mesh_file_input_h5
            
        if mesh_file.suffix == '.h5':
            with h5py.File(mesh_file, 'r') as hf:
                points = np.array(hf['Mesh']['Wall']['coordinates'])*(10**-3)
                cells = np.array(hf['Mesh']['Wall']['topology'])

                cell_type = np.ones((cells.shape[0], 1), dtype=int) * 3
                cells = np.concatenate([cell_type, cells], axis = 1)
                self.surf = pv.PolyData(points, cells)
        
        elif mesh_file.suffix == '.vtu':
            mesh = pv.read(mesh_file)
            mesh.point_data['vtkOriginalPointIds'] = np.arange(mesh.n_points)
            mask = mesh.cell_data['CellEntityIds'] == 1
            self.surf = mesh.extract_cells(mask)
            self.surf = pv.PolyData(self.surf.points*(10**-3), faces=self.surf.cells)

        return self

def partition_mesh(mesh, n_partitions, generate_global_id=False, as_composite=False): #not used
    print('Partitioning mesh')
    #yoinked this from the pyvista source code:
    alg = vtkRedistributeDataSetFilter()
    alg.SetInputData(mesh)
    alg.SetNumberOfPartitions(n_partitions)
    alg.SetPreservePartitionsInOutput(True)
    alg.SetGenerateGlobalCellIds(generate_global_id)
    alg.SetBoundaryModeToAssignToAllIntersectingRegions()
    alg.Update()

    part = alg.GetOutput()
    datasets = [part.GetPartition(ii) for ii in range(part.GetNumberOfPartitions())]
    output = pv.MultiBlock(datasets)
    if as_composite:
        return pv.merge(list(output), merge_points=False)
    return output

def initpool(arr,arr1,arr2):
    global neigh0
    global g_pts_flat 
    global invTJ_flat
    
    neigh0 = arr
    g_pts_flat = arr1
    invTJ_flat = arr2

def cell_neighbors(elems, mesh, endx):
    nodes = elems[endx, :].flatten()

    def generate_ids(i):  # numpydoc ignore=GL08
        ids = vtk.vtkIdList()
        ids.InsertNextId(i)
        return ids

    neighbors = set()
    for i in nodes.tolist():
        point_ids = generate_ids(i)
        cell_ids = vtk.vtkIdList()
        mesh.GetCellNeighbors(endx, point_ids, cell_ids)
        neighbors.update([cell_ids.GetId(i) for i in range(cell_ids.GetNumberOfIds())])
    return list(neighbors)

def get_neighbours(case_name, mesh=None, neigh0=None, shp=None, shpx=None, g_pts_flat=None, invTJ_flat=None, x=None): # neigh=None, g_pts_flat=None, invTJ_flat=None, 
    """
    Get all neighbouring elements
    """
    if not Path('./neighbour_info_{}.h5'.format(case_name)).exists():
        elems=mesh.cells.reshape(-1,5)[:,1:5]
        for endx in range(len(elems)):
            cellid = mesh.cell_data['vtkOriginalCellIds'][endx] #check this vtkGlobalCellIds
            nb0 = cell_neighbors(elems,mesh, endx) #these are the neighbours in the submesh
            nb = mesh.cell_data['vtkOriginalCellIds'][nb0] #these are the original cell ids in the big mesh
            if np.sum(np.array(neigh0[cellid*200:cellid*200+200]))>0: #if there is already data here we need to add to it
                current_array = np.array(neigh0[cellid*200:cellid*200+200]).flatten()
                new_array = nb.flatten()
                nb = pd.unique(np.concatenate((current_array,new_array)))
            padding_nb = 200-len(nb)
            pad1nb = round(padding_nb/2)
            pad2nb = padding_nb-pad1nb
            neigh0[cellid*200:cellid*200+200]=np.pad(nb, (pad1nb, pad2nb), 'edge').tolist() #will need to get unique neighbours l8r
            if endx%1000==0 and x == 0: #only print on one proc
                print('{}%'.format(round(100*endx/len(elems))))
            #save the physical quadrature points and the inverse transposed Jacobians
            npts = mesh.points[elems[endx]] #physical node points
            g_pts = compute_phys_quads(npts, shp) #4x3
            invTJ_at_quads = jacobian(npts,shpx) #4x3x3
            g_pts_flat[cellid*(4*3):cellid*(4*3)+4*3]=g_pts.ravel(order='C').tolist()
            invTJ_flat[cellid*(4*3*3):cellid*(4*3*3)+4*3*3]=invTJ_at_quads.ravel(order='C').tolist()
    else:
        file = h5py.File('neighbour_info_{}.h5'.format(case_name), 'r')
        neighbors = np.array(file['neighbours']) #non-unique cells in the patch
        g_pts = np.array(file['g_pts'])
        invTJ_at_quads = np.array(file['invTJ_at_quads'])
        return neighbors, g_pts, invTJ_at_quads

def quadrature(): #four point rule
    l1 = 0.585410196624968
    l2=0.138196601125010
    pts = l2*np.ones((4, 3))
    pts[1,0]=l1
    pts[2,1]=l1
    pts[3,2]=l1 
    weights=0.25/6
    return pts, weights

def monomial(x): #x is a 4x3 array of evaluation points
    x1=x[:,0].reshape(4,1)
    x2=x[:,1].reshape(4,1)
    x3=x[:,2].reshape(4,1)
    m = np.concatenate((np.ones((4,1))-x1-x2-x3,x1,x2,x3), axis=1)
    m_x = np.concatenate((-np.ones((3,1)), np.array([[1,0,0]]).T, np.array([[0,1,0]]).T, np.array([[0,0,1]]).T),axis=1) #the same for each quad pt (P1 elems)
    return m, m_x #4x4 (quad pointsxnode points) and 3x4 arrays (refderivsxnode points)

def quad_shape():
    qpts,_ = quadrature()
    quad_shp, quad_shpx = monomial(qpts)
    return quad_shp, quad_shpx #4x4 (quad pointsxnode points) and 3x4 arrays (refderivsxnode points)

def compute_phys_quads(npts, shp):
    #4 quads x 3 coord dirs
    pts = np.matmul(shp,npts) #4quadsx4nodes x 4nodes, 3 coords
    return pts

@njit
def jacobian(npts, shpx): #npts are the physical nodal coordinates of the cell 4x[coord1, coord2,coord3]
    #Transposed Jacobian of the cell evaluated at the quad points (end up with quad points(4)xderivdirections(3)x3components array)
    J = np.zeros((4,3,3))
    invTJ = np.zeros((4,3,3)) #inverse transposed Jacobian at each quadrature point
    for q in range(4): #loop through quad points
        J[q]=np.matmul(shpx,npts) #3refderivsx4nodes x 4nodesx3dirs = 3refdx x3physicaldx (Transposed Jacobian)
        invTJ[q,:,:] = np.linalg.inv(J[q]) #this is just the inverse Jacobian
    return invTJ #4quads x 3physdx x 3refdx

@njit
def compute_derivs_at_quads(u, node_pts, shpx):
    grad = np.zeros((4,3,3))
    #with linalg
    for q in range(4):
        grad[q,:,:]=np.matmul(shpx,u[node_pts]) #3refderivsx4nodes 4nodesx3coords  = 3 refderivsx3 coords 
    return grad #quad pointsx3 refderivsx3 coords

@njit
def compute_grads_quads(u, shpx, cells, quad_grads, invTJ_at_quads, x):
    print('Computing derivatives at physical quadrature points')
    for c in range(len(cells)):
        ref_derivs_at_quads = compute_derivs_at_quads(u,cells[c],shpx)
        derivs_at_quads = np.zeros((4,3,3)) #(derivsxcoords of u)
        for j in range(4):
            derivs_at_quads[j,:,:] = np.matmul(invTJ_at_quads[c,j,:,:],ref_derivs_at_quads[j,:,:]) #(3physdxx3refdx X 3refderivsx3coords x   = 3physical derivs inversex3coords
        quad_grads[c, :, :, :] = np.transpose(derivs_at_quads, axes=(0,2,1))
        #if c==0:
        #    print(quad_grads[c])
        if c%100000==0 and x==0:
            print('{}%'.format(round(100*c/len(cells))))
@njit
def least_squares(dd, cells,quad_grads, A_flat, P0, patches, patch_pts, P2, x1):
    #need array to store totals of gradients
    grad_totals=np.zeros((len(dd.mesh.points),9))
    #need array to store instances for averaging
    grad_instances=np.zeros((len(dd.mesh.points),1))
    skip = 10
    t3 = 0
    t6 = 0
    t9 = 0
    t12 = 0
    for c in range(0,len(cells), skip): #bundle together in skip cells at a time
        if c+skip>=len(cells):
            skip = len(cells)-c
        t1 = time.time()
        A3=np.zeros((skip*36,skip*36))
        t2 = time.time()
        t3 += t2-t1
        b3 = np.zeros((skip*36,))
        t4 = time.time()
        npts_b4 = 0
        for i, ci in enumerate(range(c,c+skip)):
            #get the quad grad values of the patch
            grads_patch = quad_grads[patches[ci]] #list of gradients at each quad point in the patch (3 coordsx3derivs)
            grads = np.concatenate((quad_grads[ci].reshape(-1,9), grads_patch.reshape(-1,9)), axis=0) # include list of gradients at each quad point in cell quad_ptsx9
            #P[:,g]=[1,x,y,z]^T
            b_fill = np.matmul(P0[ci],grads) #4monomialsxnpts x npts*9comps = 4monomialsx9 components
            #Construct the system
            A = np.zeros((36,36)) #9componentsx4monomials **2
            A[0:4,0:4] = A[4:8,4:8]=A[8:12,8:12]=A[12:16,12:16]=A[16:20,16:20]=A[20:24,20:24]=A[24:28,24:28]=A[28:32,28:32]=A[32:36,32:36]=A_flat[ci*16:ci*16+16].reshape((4,4))
            A3[i*36:i*36+36,i*36:i*36+36]=A
            b3[i*36:i*36+36]=b_fill.T.flatten()
        t5 = time.time()
        t6 += t5-t4
        t7 = time.time()
        x = scipy.sparse.linalg.lsqr(scipy.sparse.csr_matrix(A3),b3)[0] #, atol=1e-10, btol=1e-10
        coeffs = x.reshape((skip,9,4)) #cellsx9componentsx4monomials
        t8 = time.time()
        t9 += t8-t7
        #reconstruct nodal derivatives
        t10 = time.time()
        for i, ci in enumerate(range(c,c+skip)):
            #if ci==0: #check if this works for one of the quadrature pts... NOTE: this will not return exactly the gradient back...
            #    print(grads[1]) #first grad pt, 0th component
            #    print(np.matmul(P0[ci][:,1],np.transpose(coeffs[i]))) #9x4^T
            grad_node = np.matmul(P2[ci],coeffs[i,:,:].T) #nodeptsxmonomials x monomialsx9components = nodepointsxcomponents
            #add to totals for each node
            grad_totals[patch_pts[ci], :] += grad_node
            grad_instances[patch_pts[ci]] += 1 #everytime we see this node, add one to the averaging denominator
        t11 = time.time()
        t12 += t11-t10
        if c%100000==0 and x1==0:
            print('{}%'.format(round(100*c/len(cells))))
    #print ('Allocating sparse matrix took {}s, loading matrices took {}s, least squares took {}s, and reconstructing gradients took {}s'.format(t3,t6,t9,t12))
    return grad_totals, grad_instances

def precompute_A(cells_parallel, P0, A_flat): #can store instead of doing every time
    for c in cells_parallel:
        A_fill = np.zeros((4,4))
        for g in range(P0[c].shape[1]):
            #Mass matrix will be the same for every component of the gradient 
            A_fill += np.matmul(P0[c][:,g].reshape((4,1)),P0[c][:,g].reshape((1,4)))
        #Add to array
        A_flat[c*16:c*16+16]=A_fill.flatten()

def get_ls_monomials(dd,neigh, pcells, g_pts, P0, patches, patch_pts, P2):
    for i in range(len(pcells)):
        patches.append(np.unique(neigh[i].flatten())) #unique cells in neighbours
        g_pts_patch = np.concatenate((g_pts[i],g_pts[patches[i]].reshape((-1,3))), axis=0) #quadptsxcoords ALSO remembering these are in mm!!
        P0.append(np.concatenate((np.ones((1,len(g_pts_patch))), np.transpose(g_pts_patch)), axis=0)) #4monomialsxquadpts big array storing monomials at each quad point in the patch
        patch_pts.append(np.unique((pcells[neigh[i,:],:].flatten()))) #unique points in each patch
        ppts_unq=dd.mesh.points[patch_pts[i]]
        P2.append(np.concatenate(((np.ones((len(ppts_unq),1))), ppts_unq), axis=1))

def generate_wss_file(dd,grads,ts):
    J = grads.reshape((-1,3,3))
    S = 0.5 * (J + np.transpose(J, axes=(0,2,1)))
    n = dd.surf.compute_normals() 
    wpoint_ids = dd.surf.point_arrays["vtkOriginalCellIds"]
    wstrain_rate=S[wpoint_ids] #wallpointsx3x3 array
    wshear_rate=-2 * np.matmul(wstrain_rate,n)*(1-np.sum(n*n, axis=1))
    mu=0.0035
    wss = mu*wshear_rate
    with h5py.File('{}_wss_{}.h5'.format(case_name, ts), 'w') as f:
        f.create_dataset(name='gradient', data=wss)


def generate_q(grads, gradfile):
    J = grads.reshape((-1,3,3)) 
    S = np.linalg.norm(0.5 * (J + np.transpose(J, axes=(0,2,1))), axis=(1,2))
    A = np.linalg.norm(0.5 * (J - np.transpose(J, axes=(0,2,1))), axis=(1,2))
    Q=0.5*(A**2 - S**2)
    gradfile.create_dataset(name='q_criterion', data=Q)

@njit
def compute_nodal_gradients(u_files, dd, shpx, x, invTJ_at_quads, A_flat, P0, patches, patch_pts, P2, print_wss, print_q):
    cells = dd.mesh.cells.reshape(-1,5)[:,1:5] #cells are preceeded by the # of verts 
    #loop through each u file:
    for idx, uf in enumerate(u_files):
        print('Computing nodal gradients for file {} of {} on processor {}'.format(idx+1, len(u_files),x))
        #have to loop through each cell to get the gradients at all quadrature points
        u = dd(uf)
        quad_grads = np.zeros((len(cells),4,3,3))
        compute_grads_quads(u, shpx, cells, quad_grads, invTJ_at_quads,x)
        ''' Commented this because it is for serial implementation only
        if not Path('quad_grads.h5').exists():
            print('Computing gradients at the quadrature points')
            start1 = time.time()
            quad_grads = np.zeros((len(cells),4,3,3))
            compute_grads_quads(u, shpx, cells, quad_grads, invTJ_at_quads,x)
            end1 = time.time()
            print('Computed quad grads in {} min!'.format(round((end1-start1)/60)))
            #file = h5py.File('quad_grads.h5', 'w')
            #file.create_dataset(name='quad_grads', data=quad_grads)
        else:
            file = h5py.File('quad_grads.h5', 'r')
            quad_grads = np.array(file['quad_grads']) 
        '''
        #Now that we have the gradients at the quadrature points, do least squares to get fit
        #print('Beginning least squares...')
        #start2 = time.time()
        grad_totals, grad_instances = least_squares(dd, cells, quad_grads, A_flat, P0, patches, patch_pts, P2,x)
        #end2 = time.time()
        #print('Finished least squares in {} min!'.format(round((end2-start2)/60)))
        #take arithmetic average of the patches for each coindicent node
        gradient=grad_totals/grad_instances
        t = dd._get_ts(uf)
        #print('Generating gradient file...')
        with h5py.File('{}_gradients_{}.h5'.format(case_name, t), 'w') as gradfile:
            gradfile.create_dataset(name='gradient', data=gradient)
            gradfile.create_dataset(name='grad_totals', data=grad_totals)
            gradfile.create_dataset(name='grad_instances', data=grad_instances)
            if print_q:
                generate_q(gradient, gradfile)
        if print_wss:
            generate_wss_file(dd,gradient,t)
        
if __name__=="__main__":
    results=sys.argv[1] #eg. case_043_low/results/
    case_name=sys.argv[2] #eg. case_043_low
    if len(sys.argv)>3:
        print_wss = sys.argv[3]
        print_q = sys.argv[4]
    else:
        print_wss = False
        print_q = False

    dd = Dataset(Path((results + os.listdir(results)[0])))
    splits = case_name.split('_')
    seg_name = 'PTSeg'+ splits[1] +'_' + splits[-1]
    main_folder = Path(results).parents[0]
    vtu_file = Path(main_folder/ ('mesh/' + seg_name + '.vtu'))
    dd = dd.assemble_mesh().assemble_surface(mesh_file=vtu_file) 
    start = time.time()
    shp, shpx = quad_shape() #shape functions evaluated at quadrature points 4x3 and 4x3x3
    print('Shape functions calculated!')
    cells = dd.mesh.cells.reshape(-1,5)[:,1:5]
    if not Path('./neighbour_info_{}.h5'.format(case_name)).exists():
        #get neighbouring elements
        if not Path('part_mesh.vtm').exists():
            #THIS GIVES THE WRONG NUMBER OF PARTITIONS!!!! wtf... So I guess I just have to make sure n_blocks isn't>80
            part_mesh = partition_mesh(dd.mesh, 40) 
            part_mesh.save('part_mesh.vtm')
        else:
            part_mesh=pv.read('part_mesh.vtm')
        print('Finished mesh partitioning!')
        #shared memory arrays
        arr = [0]*len(cells)*200
        neigh0 = mp.Array('i',arr)
        arr1 = [0]*len(cells)*4*3
        arr2 = [0]*len(cells)*4*3*3
        g_pts_flat = mp.Array('d',arr1)
        invTJ_flat = mp.Array('d',arr2)
        print('Getting neighbours...')
        processes = [mp.Process(target= get_neighbours, args=(case_name, part_mesh[x], neigh0, shp, shpx, g_pts_flat, invTJ_flat, x)) for x in range(part_mesh.n_blocks)]
        # Run processes
        for p in processes:
            p.start()
        # Exit the completed processes
        for pi, p in enumerate(processes):
            p.join()
        neigh = np.array(neigh0).reshape((len(cells),200))
        g_pts= np.array(g_pts_flat).reshape((len(cells),4,3))
        invTJ_at_quads = np.array(invTJ_flat).reshape((len(cells),4,3,3))
        end = time.time()
        print('Finished finding neighbours in {} min!'.format(round((end-start)/60)))
        with h5py.File('neighbour_info_{}.h5'.format(case_name), 'w') as file:
            file.create_dataset(name='neighbours', data=neigh)
            file.create_dataset(name='g_pts', data=g_pts)
            file.create_dataset(name = 'invTJ_at_quads', data = invTJ_at_quads)
        end2=time.time()
        print('Wrote to file in {} min!'.format(round((end2-end)/60)))
    else:
        neigh, g_pts, invTJ_at_quads = get_neighbours(case_name)
    print('Obtained neighbours!')
    
    print('Getting monomials...')
    st = time.time()
    P0 = [] #get P for each cell in advance
    patches = []
    patch_pts=[]
    P2 = []
    get_ls_monomials(dd,neigh, cells,g_pts, P0,patches,patch_pts,P2)
    en = time.time()
    print('Finished getting monomials in {} s'.format(en-st))

    #precompute A matrices in parallel
    if not Path('./A_matrices_{}.h5'.format(case_name)).exists():
        print('Precomputing A matrices...')
        s = time.time()
        A_f0=[0]*len(cells)*16
        A_flat = mp.Array('d',A_f0)
        cells_divided = math.floor(len(cells)/nprocs1)
        cells_remain = len(cells)-nprocs1*cells_divided
        #print(cells_divided,cells_remain, len(cells))
        cells_parallel = []
        for i in range(nprocs1):
            cells_parallel.append(range(i*cells_divided,i*cells_divided+cells_divided))
        cells_parallel.append(range(nprocs1*cells_divided,nprocs1*cells_divided+cells_remain))
        processes = [mp.Process(target= precompute_A, args=(cells_parallel[x], P0, A_flat)) for x in range(nprocs)]
        # Run processes
        for p in processes:
            p.start()
        # Exit the completed processes
        for p in processes:
            p.join()
        e = time.time()
        print('Obtained A matrices in {} min!'.format(round((e-s)/60)))
        A_flat0=np.array(A_flat)
        with h5py.File('A_matrices_{}.h5'.format(case_name), 'w') as file1:
            file1.create_dataset(name='A_flat', data=A_flat0)
    else:
        with h5py.File('A_matrices_{}.h5'.format(case_name), 'r') as file:
            A_flat0 = np.array(file['A_flat'])

    #this is where we need to divide up the files:    
    print('Beginning computations...')
    start_c = time.time()
    num = math.floor(len(dd.up_files)/nprocs1)
    up_files = []
    for i in range(nprocs1):
        up_files.append(dd.up_files[i*num:(i+1)*num-1])
    up_files.append(dd.up_files[nprocs1*num:-1]) #the remaining list of files
    processes = [mp.Process(target=compute_nodal_gradients, args=(up_files[x], dd, shpx, x, invTJ_at_quads, A_flat0, P0, patches, patch_pts, P2, print_wss, print_q)) for x in range(nprocs)]
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    #Serial
    #x=0
    #compute_nodal_gradients([dd.up_files[x]], dd, shpx, x, invTJ_at_quads, A_flat0, P0, patches, patch_pts, P2, print_wss, print_q)
    end_c = time.time()
    print('Computations complete in {} min'.format(round((end_c-start_c)/60)))
    print('Total required time was {} min'.format(round((end_c-start)/60)))
    
    #compute_nodal_gradients([dd.up_files[x]], dd, shpx, x, invTJ_at_quads, A_flat0, P0, patches, patch_pts, P2, print_wss, print_q)

"""
SPR Steps:
1) Get neighbouring elements to every element
2) Get quadrature point locations for each element
3) Compute shape function derivatives at the quadrature points
4) Compute the Jacobians for each cell at the quadrature points
5) Get the derivatives of u at the quadrature points on the reference tet
6) Coordinate transformation using the inverse transposed Jacobian to obtain the derivatives of u at the quadrature points on the physical tet 
7) Get the patch of all quadrature points in all connected elements
8) Create linear system and solve least squares problem to obtain the derivatives at the nodes
    Ax=b, where A is the sparse "mass" matrix of squared monomials, b is the vector of monomial expansions at the quadrature points 
9) Reform the gradient matrix at every node in the patch based on the least squares regression x-vector, ie. the monomial expansion with coefficients x at the node point
10) Each patch now has nodal gradients that correspond to that regression, have to average coincident nodes over containing patches 
"""

