from init import *
from main_funcs import *

class DiskMap():

    def __init__(self,agn):
        self.agn = agn.replace(' ','').lower()
        self.lams = []
        self.m9 = []
        self.u_space = []

    def __repr__(self):
        return(self.agn)
    
    def __str__(self):
        return(self.agn)

    def feed_lc(self,file):
        self.file = file
        table = np.genfromtxt(file)
        self.td_ij = table[:,0]
        self.lam_ij = table[:,1]
        self.lc_ij = table[:,2]
        self.err_ij = table[:,3]
        
        self.lams = np.unique(self.lam_ij)
        self.nu = len(self.lc_ij)
        print('lightcurve imported')
        
    def feed_params(self,redshift,dist,mass,nedd,alpha,inc):
        self.z = redshift
        self.dist = dist*1e6*pc2cm
        self.m9 = mass
        if self.m9 > 1000:
            print('are you sure you gave mass in units of 1/1e9 Msun...')
        self.nedd = nedd
        self.alpha = alpha
        if inc < 0 or inc>90:
            print('please give inclination between 0 and 90 deg')
        self.inc = inc
        print('physical parameters defined')

    def feed_space(self,u_space,tp_space):
        self.tp_space = tp_space
        self.u_space = u_space

        self.tp_bins = len(self.tp_space)
        self.tp_gap =  self.tp_space[1]-self.tp_space[0]
        
        u_kl = []
        tp_kl = []
        for k in range(len(self.u_space)):
            for l in range(len(self.tp_space)):
                u_kl.append(self.u_space[k])
                tp_kl.append(self.tp_space[l])
        self.u_kl = np.array(u_kl)
        self.tp_kl = np.array(tp_kl)

        self.R_space = u_to_R(self.u_space,M9=self.m9,Alpha=self.alpha)
        self.t0_max = self.R_space[-1]/3e10/86400*np.sin(self.inc*np.pi/180)*(1+self.z)

        print('model parameters defined')

    def make_kernel(self,lam):        
        lam *= 1e-8
        kernel = []
        k = 0
        while k < len(self.u_space)-1:
            trap = 0
            u_a = self.u_space[k]
            u_b = self.u_space[k+1]
            f_a = dfintegrand(T0_u(u_a,nEdd=self.nedd,M9=self.m9,Alpha=self.alpha),u_a,Lam=lam,D=1,
                              nEdd=self.nedd,M9=self.m9,Inc=0,Z=self.z,Alpha=self.alpha)
            f_b = dfintegrand(T0_u(u_b,nEdd=self.nedd,M9=self.m9,Alpha=self.alpha),u_b,Lam=lam,D=1,
                              nEdd=self.nedd,M9=self.m9,Inc=0,Z=self.z,Alpha=self.alpha)
            trap += 0.5*(f_a+f_b)*(u_b-u_a)*1e16
            kernel.append(trap)
            k += 1
        kernel.append(trap)
        kernel = np.array(kernel)
        kernel *= 1/np.max(kernel)
        return(kernel)

    def make_T_prof(self):
        T_prof = T0_u(self.u_space,nEdd=self.nedd,M9=self.m9,Alpha=self.alpha)
        return(T_prof)

    def create_df_line(self,y,Tarray):
        lam = self.lam_ij[y]*1e-8
        td = self.td_ij[y]
        matrix_line = np.zeros((1,len(self.u_space)*len(self.tp_space)))
        for l in range(len(self.tp_space)):
            tp = self.tp_space[l]
            if tp+self.tp_gap < td-self.t0_max or tp-self.tp_gap > td + self.t0_max:
                continue
            else:
                k = 0
                while k < len(self.u_space)-1:
                    u_a = self.u_space[k]
                    u_b = self.u_space[k+1]
                    R = 0.5*(self.R_space[k]+self.R_space[k+1])
                    timecon = timefunc(td,tp,self.tp_gap,R,Inc=self.inc,Z=self.z)
                    T_a = Tarray[k*len(self.tp_space)+l]
                    T_b = Tarray[(k+1)*len(self.tp_space)+l]
                    if timecon > 0:
                        f_a = dfintegrand(T_a,u_a,lam,D=self.dist,nEdd=self.nedd,M9=self.m9,Inc=self.inc,Z=self.z,Alpha=self.alpha)
                        f_b = dfintegrand(T_b,u_b,lam,D=self.dist,nEdd=self.nedd,M9=self.m9,Inc=self.inc,Z=self.z,Alpha=self.alpha)
                        trap = 0.5*(f_a+f_b)*(u_b-u_a)*1e16*timecon
                    else:
                        trap = timecon
                    x = k*len(self.tp_space)+l
                    matrix_line[0,x] = trap
                    k += 1 
                matrix_line[0,x+len(self.tp_space)] = trap
        return(matrix_line)

    def make_smooth(self,add_t_smooth,add_u_smooth):    
        T0_matrix = np.ones([len(self.u_kl),len(self.u_kl)])
        
        for line in T0_matrix:
            line *= T0_u(self.u_kl,nEdd=self.nedd,M9=self.m9,Alpha=self.alpha)
    
        normT_mat = np.identity(len(self.tp_kl))/(T0_matrix**2)
        normT = np.zeros((len(self.tp_kl)+len(self.lams),(len(self.tp_kl)+len(self.lams))))
        normT[:normT_mat.shape[0],:normT_mat.shape[1]] = normT_mat
        smooth_matrix = normT
        
        if add_t_smooth:
            difft_mat = np.identity(len(self.tp_kl))
            for i in range(len(self.tp_kl)-1):
                if self.tp_kl[i+1] >= self.tp_kl[i]:
                    difft_mat[i,i+1] = -1       
            badrow = []
            for i in range(len(difft_mat)):
                if len(np.where(difft_mat[i,:]!=0)[0]) != 2:
                    badrow.append(i)
            difft_mat *= 1/T0_matrix
            difft_mat = np.delete(difft_mat,badrow,0)
            difft_mat = np.matmul(difft_mat.T,difft_mat) 
            
            difft = np.zeros((len(self.tp_kl)+len(self.lams),(len(self.tp_kl)+len(self.lams))))
            difft[:difft_mat.shape[0],:difft_mat.shape[1]] = difft_mat
            
            smooth_matrix += difft
        
        if add_u_smooth:
            T1_matrix = np.ones([len(self.u_kl),len(self.u_kl)])
            for i in range(len(T1_matrix)):
                line = T1_matrix[i]
                if i < len(line)-len(self.tp_space):
                    line[:] = np.sqrt(T0_u(self.u_kl[i],nEdd=self.nedd,M9=self.m9,Alpha=self.alpha) * \
                                      T0_u(self.u_kl[i+len(self.tp_space)],nEdd=self.nedd,M9=self.m9,Alpha=self.alpha))
                else:
                    line[:] = T0_u(self.u_kl[i],nEdd=self.nedd,M9=self.m9,Alpha=self.alpha)
            diffu_mat = np.identity(len(self.tp_kl))
            for i in range(len(self.tp_kl)-len(self.tp_space)):
                if self.u_kl[i+len(self.tp_space)] > self.u_kl[i]:
                    diffu_mat[i,i+len(self.tp_space)] = -1
            badrow = []
            for i in range(len(diffu_mat)):
                if len(np.where(diffu_mat[i,:]!=0)[0]) != 2:
                    badrow.append(i)
            diffu_mat *= 1/T1_matrix
            diffu_mat = np.delete(diffu_mat,badrow,0)        
            diffu_mat = np.matmul(diffu_mat.T,diffu_mat) 
            
            diffu = np.zeros((len(self.tp_kl)+len(self.lams),(len(self.tp_kl)+len(self.lams))))
            diffu[:diffu_mat.shape[0],:diffu_mat.shape[1]] = diffu_mat
            
            smooth_matrix += diffu
    
        return(smooth_matrix)

    def invert_xi_csr(self,xi,matrix_chi,smooth_matrix):
        sparse_matrix = csr_matrix(matrix_chi)
        toinv = sparse_matrix.transpose().dot(sparse_matrix)+csr_matrix(xi*smooth_matrix)
        chi_one = self.lc_ij/self.err_ij
        right = sparse_matrix.transpose().dot(chi_one)
        Tmap_full = sparse.linalg.spsolve(toinv,right)
        return(Tmap_full)

    def do_ridge_pool(self,pool,xi_list,do_t_smooth=True,do_u_smooth=True):

        if not os.path.exists('./models'):
            os.makedirs('./models')    
        
        path_to_models = './models/'+self.agn
        if not os.path.exists(path_to_models):
            os.makedirs(path_to_models)       
            
        if type(xi_list) is not list:
            return('error!!!! feed me xis in list form!')

        if len(self.lams) == 0:
            return('error!!!! feed me a lightcure first with feed_lc!')
        elif len(self.u_space) == 0:
            return('error!!!! feed me model parameters first with feed_space!')
        elif type(self.m9) == list:
            return('error!!!! feed me physical parameters first with feed_params!')
            
        run_list = range(0,len(self.lam_ij))
        Tinit = T0_u(self.u_kl,nEdd=self.nedd,M9=self.m9,Alpha=self.alpha)
        list_of_results = pool.starmap(self.create_df_line,zip(run_list,repeat(Tinit)))
        matrix = np.concatenate(list_of_results)

        print('initial matrix made')
                  
        err_diag = np.diag(1/self.err_ij)
        
        matrix_add = []
        for lam in self.lams:
            matrix_add.append((self.lam_ij == lam)*1)
        matrix_add = np.array(matrix_add).T
        matrix = np.concatenate((matrix,matrix_add),axis=1)    
        matrix_chi = np.matmul(err_diag,matrix)
    
        smooth_matrix = self.make_smooth(do_t_smooth,do_u_smooth)

        print('smoothing matrix made')
    
        self.Tmap_dict = {}
        self.lc_pred_dict = {}
        self.constants_dict = {}

        print('now for some matrix inversions ... this may take a minute ...')
        for xi in xi_list:
            Tmap_full = self.invert_xi_csr(xi,matrix_chi,smooth_matrix)
            Tmap = Tmap_full[:-len(self.lams)]
            constants = Tmap_full[-len(self.lams):]
            Tmap_file = open(path_to_models+'/tp_'+str(int(self.tp_bins))+'_xi_'+str(int(xi))+'_Tmap.txt','w')
            np.savetxt(Tmap_file,Tmap)
            Tmap_file.close()

            self.Tmap_dict[xi] = Tmap
            self.constants_dict[xi] = constants
            
            lc_pred = np.matmul(matrix,Tmap_full)
            lc_pred_file = open(path_to_models+'/tp_'+str(int(self.tp_bins))+'_xi_'+str(int(xi))+'_lc.txt','w')
            np.savetxt(lc_pred_file,lc_pred)
            lc_pred_file.close()

            self.lc_pred_dict[xi] = lc_pred
            print('xi = '+str(xi)+' done!')

        print('all done')
        return()
