import sys
from pathlib import Path
import numpy as np
import vtk as _vtk
import math
import pyvista as pv
import h5py
import scipy.linalg as la

class Dataset():
    """ Load BSL-specific data and common ops. 
    """
    def __init__(self, folder, file_glob_key=None, mesh_glob_key=None):
        """ Init the dataset.

        Args:
            folder (path): a folder with h5 data and mesh files from the BSL solver.
            file_glob_key (str): key for globbing h5 files. 
            mesh_glob_key (str): key for globbing mesh h5 file.
        """
        self.folder = Path(folder)

        if mesh_glob_key is None:
            mesh_glob_key = '*h5'

        wss_folder = (folder / 'wss_files')
        if wss_folder.exists():
            wss_glob_key = '*_curcyc_*wss.h5'
            self.wss_files = sorted(wss_folder.glob(wss_glob_key),
                key=self._get_ts)

    def _get_ts(self, h5_file):
        """ Given a simulation h5_file, get ts. """
        return int(h5_file.stem.split('_ts=')[1].split('_')[0])
    
    def assemble_surface(self, mesh_file):
        """ Create PolyData from h5 mesh file. 

        Args:
            mesh_file
        """
        # assert self.mesh_file.exists(), 'mesh_file does not exist.'
        if mesh_file.suffix == '.h5':
            with h5py.File(mesh_file, 'r') as hf:
                points = np.array(hf['Mesh']['Wall']['coordinates'])
                cells = np.array(hf['Mesh']['Wall']['topology'])

                cell_type = np.ones((cells.shape[0], 1), dtype=int) * 3
                cells = np.concatenate([cell_type, cells], axis = 1)
                self.surf = pv.PolyData(points, cells)
        return self

def get_wss(wss_file, array='wss'):
    if array == 'wss':
        with h5py.File(wss_file, 'r') as hf:
            val = np.array(hf['Computed']['wss'])
    else:
        with h5py.File(wss_file, 'r') as hf:
            val = np.array(hf[array])
    return val

def Poincare_n(dd, points, cells, E, eps=0):
    #by cells - remember numbering for cells is 1-3 not 0-2
    wss = dd.surf.point_arrays['wss']

    dist1 =np.sqrt((points[cells[:,1],0]-points[cells[:,2],0])**2+(points[cells[:,1],1]-points[cells[:,2],1])**2+(points[cells[:,1],2]-points[cells[:,2],2])**2)
    a = points[cells[:,2],:]-points[cells[:,1],:] #a for each cell
    b = points[cells[:,3],:]-points[cells[:,1],:] #c for each cell
    c = np.cross(a,b)/np.concatenate((dist1.reshape(-1,1),dist1.reshape(-1,1),dist1.reshape(-1,1)), axis=1)

    #perturbed wss = wss-(wss)_n*c which is the wss minus the wss in the normal direction to each cell by the BAC-CAB rule
    v1 = np.cross(np.cross(wss[cells[:,1],:],c),c) 
    v2 = np.cross(np.cross(wss[cells[:,2],:],c),c)
    v3 = np.cross(np.cross(wss[cells[:,3],:],c),c)

    #determinant identity instead of doing this in loop
    s1 = np.sum(np.cross(c,v2)*v3, axis=1)
    s2 = np.sum(np.cross(v1,c)*v3, axis=1)
    s3 = np.sum(np.cross(v1,v2)*c, axis=1)

    #don't use loop here either
    P_pos = np.where((s1>eps) & (s2>eps) & (s3>eps), 1,0) #all have same positive sign = 1
    P_neg = np.where((s1<eps) &(s2<eps) & (s3<eps), -1,0) #all have same negative sign = -1

    P = P_pos+P_neg
    P_ndx = np.where(((s1>eps) & (s2< eps)) | ((s2>eps) & (s1< eps)), 0, P)
    
    return P_ndx #Poincare index is by the cell

def WSSDivergence(dd, case_name, cpos, wss_files):
    points = dd.surf.points
    cells = dd.surf.faces.reshape(-1,4) #preceeded by # verts
    #numb = dd.surf.n_cells
    #print(numb, cells.shape)
    N = len(points)
    E = len(cells)
    
    #cell areas and normals
    #surf = surf.compute_cell_sizes(length=False, volume=False)
    #surf = surf.compute_normals()
    Poin_stable_node = np.zeros((N,))
    Poin_unstable_node = np.zeros((N,))
    Poin_stable_focus = np.zeros((N,))
    Poin_unstable_focus = np.zeros((N,))

    for i, wss_file in enumerate(wss_files):
        ts = dd._get_ts(wss_file)
        file_old = './dynamics/{}/divWSSNorm_PoinIndex_{}.h5'.format(case_name, ts)
        if Path(file_old).exists():
            #read h5 file
            dd.surf.point_arrays['wss'] = get_wss(file_old, 'WSS')
            dd.surf.point_arrays['wss_mag'] = np.linalg.norm(dd.surf.point_arrays['wss'], axis=1)
            dd.surf.point_arrays['Poin'] = get_wss(file_old, 'Poin')
            dd.surf.point_arrays['wss_n'] = dd.surf.point_arrays['wss']/np.concatenate((dd.surf.point_arrays['wss_mag'].reshape(-1,1),dd.surf.point_arrays['wss_mag'].reshape(-1,1),dd.surf.point_arrays['wss_mag'].reshape(-1,1)), axis=1)
            dd.surf.point_arrays['div_wss'] = get_wss(file_old, 'DivWSS')
            Poin_stable_node = get_wss(file_old, 'Poin_sn')
            Poin_unstable_node = get_wss(file_old, 'Poin_un')
            Poin_stable_focus = get_wss(file_old, 'Poin_sf')
            Poin_unstable_focus = get_wss(file_old, 'Poin_uf')
            Poin_saddle_mask = get_wss(file_old, 'Poin_sp')
        else:
            #compute normalized wss
            dd.surf.point_arrays['wss'] = get_wss(wss_file)
            dd.surf.point_arrays['wss_mag'] = np.linalg.norm(dd.surf.point_arrays['wss'], axis=1)
            dd.surf.point_arrays['wss_n'] = dd.surf.point_arrays['wss']/np.concatenate((dd.surf.point_arrays['wss_mag'].reshape(-1,1),dd.surf.point_arrays['wss_mag'].reshape(-1,1),dd.surf.point_arrays['wss_mag'].reshape(-1,1)), axis=1)
            threshold=np.percentile(dd.surf.point_arrays['wss_mag'],50)
            #compute gradients
            grad = dd.surf.compute_derivative(scalars="wss", gradient=True, qcriterion=False, faster=False)
            dd.surf.point_arrays['div_wss'] = grad.point_arrays['gradient'][:,0]+grad.point_arrays['gradient'][:,5]+grad.point_arrays['gradient'][:,8]
            #Jacobian matrices
            J = grad.point_arrays['gradient'].reshape(-1,3,3)

            #compute Poincare index
            P = Poincare_n(dd, points, cells, E)
            #print('completed Poincare normals')
            #Cast Poincare index to point array
            dd.surf.point_arrays['Poin'] = np.zeros((N,))
            
            for k in range(N): 
            #    if k%100==0:
            #        print(k)
                mask = np.where((cells[:, 1] == k) | (cells[:, 2] == k) | (cells[:, 3] == k), True, False) 
                if np.min(P[mask])==0:
                    dd.surf.point_arrays['Poin'][k]=np.max(P[mask])    
                else:
                    dd.surf.point_arrays['Poin'][k]=np.min(P[mask])  

                if dd.surf.point_arrays['wss_mag'][k] >= threshold:
                    dd.surf.point_arrays['Poin'][k]= 0
                #get eigenvalue classification
                if (dd.surf.point_arrays['Poin'][k]>0):
                    ev = la.eigvals(J[k,:,:])
                    comp = np.iscomplex(ev).sum()
                    if comp > 1:
                        if ev[0]>0:
                            Poin_unstable_focus[k] = 1
                        else:
                            Poin_stable_focus[k] = 1
                    else:
                        if (ev[0]<ev[1]) and (ev[1]<ev[2]) and (ev[2]<0):
                            Poin_stable_node[k]=1
                        elif (ev[0]>ev[1]) and (ev[1]>ev[2]) and (ev[2]>0):
                            Poin_unstable_node[k]=1

            #write to h5
            f = h5py.File('./dynamics/{}/divWSSNorm_PoinIndex_{}.h5'.format(case_name, ts), 'w')
            f.create_dataset('DivWSS', data = dd.surf.point_arrays['div_wss'])
            f.create_dataset('WSS', data = dd.surf.point_arrays['wss'])
            f.create_dataset('Poin', data = dd.surf.point_arrays['Poin'])
            f.create_dataset('Poin_uf', data = Poin_unstable_focus)
            f.create_dataset('Poin_sf', data = Poin_stable_focus)
            f.create_dataset('Poin_un', data = Poin_unstable_node)
            f.create_dataset('Poin_sn', data = Poin_stable_node)

            
            Poin_saddle_mask = np.where(dd.surf.point_arrays['Poin'] == 1, 1, 0)
            f.create_dataset('Poin_sp', data = Poin_saddle_mask)
        #make an animation
        p = pv.Plotter(off_screen=True, window_size=[768,768])
        p.camera_position = cpos
        p.add_mesh(dd.surf, scalars='div_wss', cmap = 'bwr', smooth_shading=True, log_scale=True)#scalars = 'wss_mag')
        arrows = dd.surf.glyph(scale=False, orient="wss_n", tolerance = 0.00001, factor=0.3, geom=pv.Arrow())#tolerance = 0.001, factor=0.9
        p.add_mesh(arrows, color='black')
        if np.abs(np.sum(dd.surf.points[Poin_saddle_mask==1]))>0:
            p.add_points(dd.surf.points[Poin_saddle_mask==1], render_points_as_spheres=True, color = 'yellow', point_size=10, label='saddle points')
        if np.abs(np.sum(dd.surf.points[Poin_unstable_focus==1]))>0:
            p.add_points(dd.surf.points[Poin_unstable_focus==1], render_points_as_spheres=True, color = 'red', point_size=10, label='unstable focus')
        if np.abs(np.sum(dd.surf.points[Poin_stable_focus==1]))>0:
            p.add_points(dd.surf.points[Poin_stable_focus==1], render_points_as_spheres=True, color = 'green', point_size=10, label='stable focus')
        if np.abs(np.sum(dd.surf.points[Poin_unstable_node==1]))>0:
            p.add_points(dd.surf.points[Poin_unstable_node==1], render_points_as_spheres=True, color = 'blue', point_size=10, label='unstable node')
        if np.abs(np.sum(dd.surf.points[Poin_stable_node==1]))>0:
            p.add_points(dd.surf.points[Poin_stable_node==1], render_points_as_spheres=True, color = 'white', point_size=10, label='stable node')   
        legend_entries=[['saddle points', 'yellow'],['unstable focus', 'red'],['stable focus', 'green'],['unstable node', 'blue'],['stable node', 'white']]
        p.add_legend(legend_entries )     
        p.show(screenshot='dynamics/{}/imgs/fixed_points_{}'.format(case_name, ts), auto_close=False)
        print('plotted')
        actors = [x for x in p.renderer.actors.keys()]
        for a in actors:
            p.remove_actor(actors)
    p.close()
    #write the tec file
    '''
    f = open('./dynamics/divWSSNorm_PoinIndex_%i.tec'%i, 'a')
    head = 'VARIABLES = X,Y,Z, WSS, divWSS, PoincareI\nZONE N={}, E={}'.format(str(N), str(E)) + ', F=FEPOINT,ET=TRIANGLE'
    coord=np.concatenate((points[:,0].reshape(-1,1), points[:,1].reshape(-1,1), points[:,2].reshape(-1,1), dd.surf.point_arrays['wss'].reshape(-1,1), dd.surf.point_arrays['div_wss'].reshape(-1,1), Poin), axis=1)
    np.savetxt(f, coord, header = head, fmt = '%f', comments='')
    f.write("\n")
    np.savetxt(f, cells, fmt='%i')
    f.close()
    '''
if __name__ == "__main__":
    results_folder = Path(sys.argv[1])
    case_name = sys.argv[2]
    dd = Dataset(results_folder)
    main_folder = Path(results_folder).parents[1]
    #vtu_file = Path(main_folder/ ('mesh/' + case_name + '.vtu'))
    #dd = dd.assemble_surface(mesh_file=vtu_file) 
    h5_file = Path(main_folder/ ('data/' + case_name + '.h5'))
    dd=dd.assemble_surface(mesh_file=h5_file)
    imgs_folder = Path(('dynamics/{}/imgs'.format(sys.argv[2])))
    if not imgs_folder.exists():
        imgs_folder.mkdir(parents=True, exist_ok=True)
    s = sys.argv[3]
    e = sys.argv[4]
    print('solving files {} to {}'.format(s,e))
    wss_files = dd.wss_files[int(s):int(e)]
    cpos = [(17.649447468732703, -25.625457981316902, 20.378326781069894), (-2.8439961369396833, -10.443120909352007, 9.187968669868942), (-0.3328094736762078, 0.22562571892084304, 0.9156041116076414)]
    WSSDivergence(dd, case_name, cpos, wss_files)