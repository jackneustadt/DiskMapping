import numpy as np
import glob,os
from import_lc import import_lc
import matplotlib.pyplot as plt
import time
from multiprocessing import Process, Pool, active_children
import scipy
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from itertools import repeat

c = 2.998e10
G = 6.672e-8
h = 6.626e-27
kB = 1.381e-16
Msun = 1.989e33
pc2cm = 3.086e18

######### defining a bunch of functions ###################

def T0_u(u,nEdd,M9,Alpha):
    Tin = (1.529e22 * (nEdd)/M9 /(Alpha/2)**3)**0.25 #1.518 if calculated by hand
    val = Tin * u**(-0.75) * (1-u**-0.5)**0.25
    return(val)

def dBdT_lam(T,lam):
    x = h*c/lam/kB/T
    val = x * np.exp(x) / (np.exp(x)-1)**2 / T
    return(val)

def B_lam(T,lam):
    x = h*c/lam/kB/T
    val = (np.exp(x)-1)**-1
    return(val)

def Flam0(lam,D,M9,Inc,Alpha):
    Rin = G*Msun*1e9*M9/c**2*Alpha
    val = 4*np.pi*h * c**2 * np.cos(Inc*np.pi/180) * Rin**2 /lam**5/D**2 * 1e-8 
    return(val)

def u_to_R(u,M9,Alpha):
    Rin = G*Msun*1e9*M9/c**2*Alpha
    val = u*Rin
    return(val)

def R_to_u(R,M9,Alpha):
    Rin = G*Msun*1e9*M9/c**2*Alpha
    val = R/Rin
    return(val)

def fintegrand(T,u,lam,D,nEdd,M9,Inc,Z,Alpha):
    val = B_lam(T,lam/(1+Z))*Flam0(lam/(1+Z),D,M9=M9,Inc=Inc,Alpha=Alpha)*u*(1+Z)
    return(val)    

def dfintegrand(T,u,lam,D,nEdd,M9,Inc,Z,Alpha):
    val = dBdT_lam(T,lam/(1+Z))*Flam0(lam/(1+Z),D,M9,Inc,Alpha)*u*(1+Z)
    return(val)

#just to check
# def cintegrand(u,lam):
#     Tc = T0_u(u,nEdd=0.2,M9=m9,Alpha=alpha)
#     val = fintegrand(u=u,lam=lam,T=Tc,D=dist,nEdd=0.2,M9=m9,Inc=inc,Z=z,Alpha=alpha) 
#     return(val)

def G1(t1,t2,tp_gap,td,t0):
    if t2 < t1:
        val = 0
        return(val)
    t1 -= td
    t2 -= td
    t1 = round(t1,6)
    t2 = round(t2,6)
    t0 = round(t0,6)
    val = 1/np.pi/tp_gap * ((t0**2 - t2**2)**0.5 - (t0**2 - t1**2)**0.5) #version in PAPER is wrong, this is right
    return(val)

def G2(t1,t2,tp_gap,td,t0):
    if t2 < t1:
        val = 0
        return(val)
    t1 -= td
    t2 -= td
    t1 = round(t1,8)
    t2 = round(t2,8)
    t0 = round(t0,8)
    val = 1/np.pi * (np.arcsin(t2/t0) - np.arcsin(t1/t0 ))
    return(val)

def timefunc_long(td,tp,tp_gap,R,Inc,Z):
    t0 = R/3e10/86400*np.sin(Inc*np.pi/180)*(1+Z)
    t1 = np.max([tp,td-t0]) ### t1,t2 in this code are t3,t4 in the paper
    t2 = np.min([tp+tp_gap,td+t0]) 
    t3 = np.max([tp-tp_gap,td-t0]) 
    t4 = np.min([tp,td+t0]) 
    val1 = G1(t1,t2,tp_gap,td,t0) + (tp+tp_gap-td)/tp_gap*G2(t1,t2,tp_gap,td,t0) 
    val2 = -1*G1(t3,t4,tp_gap,td,t0) + (td-tp+tp_gap)/tp_gap*G2(t3,t4,tp_gap,td,t0) #this version is right 
    val = val1+val2
    return(val)

############# initializing stuff ################
z = 0.031455
dist =136*1e6*pc2cm
m9 = 0.0385
nedd = 0.14
alpha = 6
inc = 30

agn= 'Mrk 817'

lc_ij,err_ij,td_ij,lam_ij = import_lc(agn)

u_space = np.logspace(0.01,3,num=50)

tp_bins = 250

######### filtering outliers ###################

lams = np.unique(lam_ij)

nu = np.abs(len(lc_ij))

####### generate matrix ###############

tp_space = np.linspace(np.min(td_ij)-0.00001,np.max(td_ij)+0.00001,num=tp_bins)

u_kl = []
tp_kl = []

for k in range(len(u_space)):
    for l in range(len(tp_space)):
        u_kl.append(u_space[k])
        tp_kl.append(tp_space[l])

u_kl = np.array(u_kl)
tp_kl = np.array(tp_kl)

R_space = u_to_R(u_space,M9=m9,Alpha=6)
tp_gap = tp_space[1]-tp_space[0]
t0_max = R_space[-1]/3e10/86400*np.sin(inc*np.pi/180)*(1+z)

def create_df_line(y,params):
    lam = lam_ij[y]*1e-8
    td = td_ij[y]
    matrix = np.zeros((1,len(u_kl)))
    for l in range(len(tp_space)):
        tp = tp_space[l]
        if tp+tp_gap < td-t0_max or tp-tp_gap > td + t0_max:
            continue
        else:
            k = 0
            while k < len(u_space)-1:
                u_a = u_space[k]
                u_b = u_space[k+1]
                R = 0.5*(R_space[k]+R_space[k+1])
                timecon = timefunc_long(td,tp,tp_gap,R,Inc=inc,Z=z)
                T_a = params[k*len(tp_space)+l]
                T_b = params[(k+1)*len(tp_space)+l]
                if timecon > 0:
                    f_a = dfintegrand(T_a,u_a,lam,D=dist,nEdd=nedd,M9=m9,Inc=inc,Z=z,Alpha=alpha)
                    f_b = dfintegrand(T_b,u_b,lam,D=dist,nEdd=nedd,M9=m9,Inc=inc,Z=z,Alpha=alpha)
                    trap = 0.5*(f_a+f_b)*(u_b-u_a)*1e16*timecon
                else:
                    trap = timecon
                x = k*len(tp_space)+l
                matrix[0,x] = trap
                k += 1 
            matrix[0,x+len(tp_space)] = trap
    return(matrix)

def integrate_lightcurve(y,params):
    lam = lam_ij[y]*1e-8
    td = td_ij[y]
    trap = 0
    for l in range(len(tp_space)):
        tp = tp_space[l]
        if tp+tp_gap < td-t0_max or tp-tp_gap > td + t0_max:
            continue
        else:
            k = 0
            while k < len(u_space)-1:
                u_a = u_space[k]
                u_b = u_space[k+1]
                R = 0.5*(R_space[k]+R_space[k+1])
                timecon = timefunc_long(td,tp,tp_gap,R,Inc=inc,Z=z)
                T_a = params[k*len(tp_space)+l]
                T_b = params[(k+1)*len(tp_space)+l]
                if timecon > 0:
                    f_a = fintegrand(T_a,u=u_a,lam=lam,D=dist,nEdd=nedd,M9=m9,Inc=inc,Z=z,Alpha=alpha)
                    f_b = fintegrand(T_b,u=u_b,lam=lam,D=dist,nEdd=nedd,M9=m9,Inc=inc,Z=z,Alpha=alpha)
                    trap_add = 0.5*(f_a+f_b)*(u_b-u_a)*1e16*timecon
                else:
                    trap_add = 0
                trap += trap_add
                k += 1 
            trap += trap_add
    index, = np.where(lams==lam_ij[y])
    index = index[0]-len(lams)
    trap += params[index]
    return(trap)

def synthesize_lightcurve(params):
    lam_synth = []
    lc_synth = []
    td_synth = []
    for lam in lams:
        for l in range(len(tp_space)):
            tp = tp_space[l]        
            trap = 0
            k = 0
            while k < len(u_space)-1:
                u_a = u_space[k]
                u_b = u_space[k+1]
                T_a = params[k*len(tp_space)+l]
                T_b = params[(k+1)*len(tp_space)+l]
                f_a = fintegrand(T_a,u=u_a,lam=lam*1e-8,D=dist,nEdd=nedd,M9=m9,Inc=inc,Z=z,Alpha=alpha)
                f_b = fintegrand(T_b,u=u_b,lam=lam*1e-8,D=dist,nEdd=nedd,M9=m9,Inc=inc,Z=z,Alpha=alpha)
                trap += 0.5*(f_a+f_b)*(u_b-u_a)*1e16
                k += 1 
            trap += 0.5*(f_a+f_b)*(u_b-u_a)*1e16
            index, = np.where(lams==lam)
            index = index[0]-len(lams)
            trap += params[index]
            lam_synth.append(lam)
            td_synth.append(tp)
            lc_synth.append(trap)
    lc_synth = np.array(lc_synth)
    lam_synth = np.array(lam_synth)
    td_synth = np.array(td_synth)
    return(lc_synth,td_synth,lam_synth)


def invert_lamb_csr(lamb,matrix_chi,smooth_matrix,lc_ij,lc_predict):
    sparse_matrix = csr_matrix(matrix_chi)
    toinv = sparse_matrix.transpose().dot(sparse_matrix)+csr_matrix(lamb*smooth_matrix)
    chi_one = (lc_ij-lc_predict)/err_ij
    right = sparse_matrix.transpose().dot(chi_one)
    params_full = sparse.linalg.spsolve(toinv,right)
    return(params_full)
    
def make_smooth():    
    normT = np.identity(len(tp_kl))
    T_diag = np.diag(1/T0_u(u_kl,nedd,m9,alpha)**2)
    normT_mat = np.matmul(T_diag,normT)
    normT = np.zeros((len(tp_kl)+len(lams),(len(tp_kl)+len(lams))))
    normT[:normT_mat.shape[0],:normT_mat.shape[1]] = normT_mat
    smooth_matrix = normT
    return(smooth_matrix)

def do_job(pool,lamb,params,lc_ij):
    start = time.time()

    run_list = range(0,len(lam_ij))
    list_of_results = pool.starmap(create_df_line,zip(run_list,repeat(params)))
    matrix = np.concatenate(list_of_results)
    end_init = time.time()
#     print(end_init-start)
                                   
    list_of_results = pool.starmap(integrate_lightcurve,zip(run_list,repeat(params)))
    lc_predict = np.array(list_of_results)
                          
    end_integrate = time.time()

#     print(end_integrate-end_init)

    err_diag = np.diag(1/err_ij)
    
    matrix_add = []
    for lam in lams:
        matrix_add.append((lam_ij == lam)*1)
    matrix_add = np.array(matrix_add).T
    matrix = np.concatenate((matrix,matrix_add),axis=1)    
    matrix_chi = np.matmul(err_diag,matrix)

    smooth_matrix = make_smooth()
    end_mat = time.time()

#     print(end_mat-end_init)

    start_matmul = time.time()
    
    params_new = invert_lamb_csr(lamb,matrix_chi,smooth_matrix,lc_ij,lc_predict)
   
    end_matmul = time.time()
    
#     print(end_matmul-start_matmul)
    
    return(params_new)

# lc_dummy = np.array(lc_ij) 
# mean_lc_array = []
# for lam in lams:
#     index, = np.where(lam == lam_ij)
#     mean_lc = np.mean(lc_dummy[index])
#     lc_dummy[index] = mean_lc
#     mean_lc_array.append(0.5*mean_lc)
    
# params_init = T0_u(u_kl,nEdd=nedd,M9=m9,Alpha=alpha)
# params_init = np.concatenate((params_init,np.array(mean_lc_array)))
# lamb_init = 1000

# pool = Pool(processes=15)
# params_new = params_init + do_job(pool,lamb_init,params_init,lc_dummy)
# index, = np.where(params_new[:-len(lams)] < 100)
# params_new[index] = 100
# # index, = np.where(params_new < 0)
# # params_new[index] = 0
# run_list = range(0,len(lam_ij))
# list_of_results = pool.starmap(integrate_lightcurve,zip(run_list,repeat(params_new)))
# lc_predict = np.array(list_of_results)
# chi2_old = np.sum((lc_ij-lc_predict)**2/err_ij**2)/nu
# print(round(chi2_old,3))

# run_length = 31
# # lamb = lamb_init

# # lamb_range = [10000]*100

# lamb = 1000

# for i in range(run_length):
#     params_old = params_new
#     params_new = params_old + do_job(pool,lamb,params_old,lc_ij)
#     index, = np.where(params_new[:-len(lams)] < 100)
#     params_new[index] = 100
# #     index, = np.where(params_new < 0)
# #     params_new[index] = 0
#     list_of_results = pool.starmap(integrate_lightcurve,zip(run_list,repeat(params_new)))
#     print('run '+str(i+1)+' done')
#     lc_predict = np.array(list_of_results)
#     chi2_new = np.sum((lc_ij-lc_predict)**2/err_ij**2)/nu
#     print(round(chi2_new,3))
#     print(params_new[-len(lams):])
    
#     if chi2_old - chi2_new < 0:
#         lamb *= 1.5
#         params_new = params_old
#         chi2_new = chi2_old
#         print('increasing lambda, new lambda = '+str(int(lamb)))
#         print('using previous params')

#     elif chi2_old - chi2_new < 0.3:
#         lamb *= 0.5
#         print('shrinking lambda, new lambda = '+str(int(lamb)))
#         chi2_old = chi2_new

#     else:
#         chi2_old = chi2_new
        
#     fig = plt.figure(figsize=(25,15))
#     k = 1
#     for lam in lams:
#         index, = np.where(lam == lam_ij)
#         ax = plt.subplot(int(np.ceil(len(lams)/4)),4,k)
#         ax.errorbar(td_ij[index],lc_ij[index],yerr=err_ij[index],marker='.',color='b',ls='none')
#         ax.plot(td_ij[index],lc_predict[index],'.r')
#         ax.text(np.mean(td_ij),np.min(lc_ij[index]),str(lam),fontsize=15)

#         ax.plot(td_ij[index],lc_predict[index],'-g')

#         ax.set_xlim(np.min(td_ij)-10,np.max(td_ij)+10)

#         k+=1
#     plt.tight_layout()
#     plt.savefig('../panels/LM/tp_'+str(int(tp_bins))+'_lc_run_'+str(i+1)+'.png',
#                 format='png',transparent=False,facecolor='w')
#     plt.close()
    
#     params_trim = params_new[:-len(lams)]
#     params_plot = np.reshape(params_trim,(-1,len(tp_space)))

#     mean_temp = []
#     for line in params_plot:
#         mean_temp.append(np.median(line))
#     mean_temp = np.array(mean_temp)

#     frac_plot = np.matmul(np.diag(1/mean_temp),params_plot)
#     for line in frac_plot:
#         line += -1*np.median(line)

#     sorted_array = sorted(np.abs(frac_plot.flatten()))
#     scalar = np.max(sorted_array)

#     vmax = np.max(scalar)
#     vmin = -1*vmax

#     plt.figure(figsize=(8,6))
#     plt.contourf(tp_space,u_space,frac_plot,15,cmap='coolwarm',vmin=vmin,vmax=vmax)
#     plt.colorbar(label=r'$\Delta T$ (K, temp)',)
#     plt.ylim(1,1e3)

#     plt.yscale('log')
#     plt.title(r'$\lambda=$'+f"{lamb:.1e}"+r', $\chi^2=$'+f"{chi2_old:.3f}"+r', scale ='+f"{scalar:.3f}")
#     plt.ylabel('u (R/Rin, radius)')
#     plt.xlabel('MJD (time)')
#     plt.savefig('../panels/LM/tp_'+str(int(tp_bins))+'_model_run_'+str(i+1)+'.png',
#                 format='png',transparent=False,facecolor='w')
#     plt.close()
    

# params_file = open('../panels/LM/tp_'+str(int(tp_bins))+'_end_model.txt','w')
# np.savetxt(params_file,params_new)
# params_file.close()
# plt.close()    
    
params = np.genfromtxt('../panels/LM/tp_'+str(int(tp_bins))+'_end_model.txt')
params_trim = params[:-len(lams)]
params_plot = np.reshape(params_trim,(-1,len(tp_space)))

mean_temp = []
for line in params_plot:
    mean_temp.append(np.median(line))
mean_temp = np.array(mean_temp)

frac_plot = np.matmul(np.diag(1/mean_temp),params_plot)
for line in frac_plot:
    line += -1*np.median(line)

sorted_array = sorted(np.abs(frac_plot.flatten()))
scalar = sorted_array[int(len(sorted_array)*0.99)]
scalar = np.max(sorted_array)

vmax = np.max(scalar)
vmin = -1*vmax

plt.figure(figsize=(8,6))
plt.contourf(tp_space,u_space,frac_plot,15,cmap='coolwarm',vmin=vmin,vmax=vmax)
plt.colorbar(label=r'$\Delta T$ (K, temp)',)
plt.ylim(1,1e3)

# lamb = 240
# chi2_old = 4.806
plt.yscale('log')
# plt.title(r'$\lambda=$'+f"{lamb:.1e}"+r', $\chi^2=$'+f"{chi2_old:.3f}"+r', scale ='+f"{scalar:.3f}")
plt.ylabel('u (R/Rin, radius)')
plt.xlabel('MJD (time)')
plt.savefig('../panels/LM/tp_'+str(int(tp_bins))+'_model.png',
            format='png',transparent=False,facecolor='w')

plt.figure(figsize=(8,6))
plt.plot(u_space,mean_temp)
plt.plot(u_space,T0_u(u_space,nEdd=nedd,M9=m9,Alpha=alpha))
plt.plot(u_space,T0_u(u_space,nEdd=0.2,M9=m9,Alpha=alpha))
plt.xscale('log')
plt.yscale('log')
plt.savefig('../panels/LM/tp_'+str(int(tp_bins))+'_test_meanT.png',
            format='png',transparent=False,facecolor='w')

#####################################################################################################

pool = Pool(processes=15)
run_list = range(0,len(lam_ij))
dummy_temp = np.concatenate((T0_u(u_kl,nEdd=nedd,M9=m9,Alpha=alpha),params[-len(lams):]))
list_of_results = pool.starmap(integrate_lightcurve,zip(run_list,repeat(dummy_temp)))
lc_predict = np.array(list_of_results)
lc_synth,td_synth,lam_synth = synthesize_lightcurve(params)
fig = plt.figure(figsize=(25,15))
k = 1
for lam in lams:
    index, = np.where(lam == lam_ij)
    ax = plt.subplot(int(np.ceil(len(lams)/4)),4,k)
    ax.errorbar(td_ij[index],lc_ij[index],yerr=err_ij[index],marker='.',color='b',ls='none')
    ax.text(np.mean(td_ij),np.min(lc_ij[index]),str(lam),fontsize=15)
    
    index2, = np.where(lam == lam_synth)
    
    ax.plot(td_synth[index2],lc_synth[index2],'-r')

#     ax.plot(td_ij[index],lc_predict[index],'-g')

    ax.set_xlim(np.min(td_ij)-10,np.max(td_ij)+10)

    k+=1
plt.tight_layout()
plt.savefig('../panels/LM/tp_'+str(int(tp_bins))+'_test_lc.png',
            format='png',transparent=False,facecolor='w')
    
#####################################################################################################
    
fig = plt.figure(figsize=(8,6))
lamk = -len(lams)

for lam in lams:
    index, = np.where(lam == lam_ij)

    plt.plot(lam,np.mean(lc_predict[index])-params[lamk],'.r')
    plt.plot(lam,np.mean(lc_ij[index]),'.b')
#     test_sum = scipy.integrate.quad(cintegrand,np.min(u_space),np.max(u_space),args=(lam*1e-8,))[0]*1e16
#     plt.plot(lam,test_sum,'ok')

    index2, = np.where(lam == lam_synth)
    plt.plot(lam,np.mean(lc_synth[index2])-params[lamk],'.g')
    if lam == lams[0]:
        plt.plot(lam,np.mean(lc_predict[index])-params[lamk],'.r',label='integrated steady state')
        plt.plot(lam,np.mean(lc_ij[index]),'.b',label='data')
    lamk+=1 
    
plt.plot(lams,9*(lams/lams[-1])**(-7/3),'-k',label='lambda**-7/3')
plt.plot(lams,0.7*np.max(lc_ij)*(lams/lams[0])**(-4/3),':k',label='lambda**-4/3')
plt.plot(lams,params[-len(lams):],'--k',label='offset')

plt.ylim(10,5000)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('wavelength')
plt.ylabel('flux [1e-16 erg s-1 cm-2 A-1]')
plt.legend()
plt.tight_layout()
plt.savefig('../panels/LM/tp_'+str(int(tp_bins))+'_test_SED.png',
            format='png',transparent=False,facecolor='w')


# #####################################################################################################

# ref_lam = 1180
# index, = np.where((ref_lam == lam_synth)&(td_synth<59650))

# ref_constant = 0
# ref_flux = lc_synth[index]-ref_constant

# fig = plt.figure(figsize=(20,15))
# k = 1

# constant_array = [0]

# for lam in lams:
#     if lam != ref_lam:
#         ax = plt.subplot(4,4,k)
        
#         index, = np.where((lam == lam_synth)&(td_synth<59650))
        
#         mask, = np.where(ref_flux<1300)
        
#         x = ref_flux[mask]
#         y = lc_synth[index][mask]
#         ax.plot(x,y,'ob',ls='none')

#         num = len(x)*np.sum(y*x)-np.sum(x)*np.sum(y)
#         den = len(x)*np.sum(x**2)-np.sum(x)**2
#         slope = num/den

#         constant = np.mean(y)-slope*np.mean(x)
#         pred = constant + slope*(x)
#         slope_err = np.sqrt(np.sum((y-pred)**2)/np.sum((x-np.mean(x))**2)/(len(x)-2))
#         constant_err = np.sqrt(np.sum((y-pred)**2)*np.sum(x**2)/len(x)/np.sum((x-np.mean(x))**2)/(len(x)-2))
        
#         ax.text(1200,np.mean(y),str(np.round(constant,2)),color='red')
#         ax.text(600,np.max(y),str(lam))
        
#         ax.plot(x,pred,'--r')
        
#         num = len(y)*np.sum(y*x)-np.sum(x)*np.sum(y)
#         den = len(x)*np.sum(y**2)-np.sum(y)**2
#         slope = num/den

#         constant = np.mean(x)-slope*np.mean(y)
#         pred = constant + slope*(y)
        
#         ax.plot(pred,y,':g')
        
#         constant = -1*constant/slope
#         ax.text(1200,np.mean(y)-0.5*np.std(y),str(np.round(constant,2)),color='green')
        
#         delta = 1/(np.std(x)**2/np.std(y)**2)
        
#         sxx = np.mean(x**2)-np.mean(x)**2
#         sxy = np.mean(x*y)-np.mean(x)*np.mean(y)
#         syy = np.mean(y**2)-np.mean(y)**2
        
#         slope = (syy-delta*sxx+np.sqrt((syy-delta*sxx)**2+4*delta*sxy**2))/(2*sxy)
#         constant = np.mean(y)-slope*np.mean(x)
#         pred = constant+slope*x
                
#         ax.plot(x,pred,'-k')
        
#         constant_array.append(constant)
        
#         ax.text(1100,np.mean(y)-1.0*np.std(y),str(np.round(constant,2)),color='black')

#         k+=1
        
# plt.tight_layout()
# plt.savefig('../panels/LM/tp_'+str(int(tp_bins))+'_test_flux_flux.png',
#             format='png',transparent=False,facecolor='w')

# #####################################################################################################
    
# fig = plt.figure(figsize=(8,6))
# k=0
# constant_array[0] = ref_constant

# def T0_u_mod(u,nEdd,M9,Alpha):
#     Tin = (1.529e22 * (nEdd)/M9 /(Alpha/2)**3)**0.25 #1.518 if calculated by hand
#     val = Tin * u**(-0.875) * (1-u**-0.5)**0.25
#     return(val)

# dummy_temp = np.concatenate((T0_u(u_kl,nEdd=0.75*nedd,M9=m9,Alpha=alpha),params[-len(lams):]))
# lc_synth,td_synth,lam_synth = synthesize_lightcurve(dummy_temp)

# for lam in lams:
#     index, = np.where(lam == lam_ij)

#     plt.plot(lam,np.mean(lc_ij[index]),'.b')
#     plt.plot(lam,np.mean(lc_ij[index])-constant_array[k],'.r')
    
#     index2, = np.where(lam == lam_synth)
#     plt.plot(lam,np.mean(lc_synth[index2])-params[k-len(lams)],'.g')


#     if lam == lams[0]:
#         plt.plot(lam,np.mean(lc_ij[index]),'.b',label='data')
#         plt.plot(lam,np.mean(lc_ij[index])-constant_array[k],'.r',label='data-constant')
#         pivot43 = np.mean(lc_ij[index])
        
#     if lam == lams[-1]:
#         pivot73 = np.mean(lc_ij[index]-constant_array[k])

#     k+=1
    
# plt.plot(lams,pivot73*(lams/lams[-1])**(-7/3),'-k',label='lambda**-7/3')
# plt.plot(lams,pivot43*(lams/lams[0])**(-4/3),':k',label='lambda**-4/3')
# plt.plot(lams,constant_array,'--k',label='constants')

# plt.ylim(10,5000)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('wavelength')
# plt.ylabel('flux [1e-16 erg s-1 cm-2 A-1]')
# plt.legend()
# plt.tight_layout()
# plt.savefig('../panels/LM/tp_'+str(int(tp_bins))+'_test_SED_constants.png',
#             format='png',transparent=False,facecolor='w')
