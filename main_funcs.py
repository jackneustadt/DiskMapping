from init import *
    
def import_lc(data_file):
    table = np.genfromtxt(data_file)
    td_ij = table[:,0]
    lam_ij = table[:,1]
    lc_ij = table[:,2]
    err_ij = table[:,3]
    return(td_ij,lam_ij,lc_ij,err_ij)

def T0_u(u,nEdd,M9,Alpha):
    Tin = (1.529e22 * (nEdd)/M9 /(Alpha/2)**3)**0.25 #1.518 if calculated by hand
    val = Tin * u**(-0.75) * (1-u**-0.5)**0.25
    return(val)

def B_lam(T,Lam):
    x = h*c/Lam/kB/T
    val = (np.exp(x)-1)**-1
    return(val)

def dBdT_lam(T,Lam):
    x = h*c/Lam/kB/T
    val = x * np.exp(x) / (np.exp(x)-1)**2 / T
    return(val)

def Flam0(lam,D,M9,Inc,Alpha):
    Rin = G*Msun*1e9*M9/c**2*Alpha
    val = 4*np.pi*h * c**2 * np.cos(Inc*np.pi/180) * Rin**2 /lam**5/D**2 * 1e-8 *0.5
    return(val)

def u_to_R(u,M9,Alpha):
    Rin = G*Msun*1e9*M9/c**2*Alpha
    val = u*Rin
    return(val)

def R_to_u(R,M9,Alpha):
    Rin = G*Msun*1e9*M9/c**2*Alpha
    val = R/Rin
    return(val)
   
def fintegrand(T,u,Lam,D,nEdd,M9,Inc,Z,Alpha):
    val = B_lam(T,Lam/(1+Z))*Flam0(Lam/(1+Z),D,M9=M9,Inc=Inc,Alpha=Alpha)*u*(1+Z)
    return(val)    

def dfintegrand(T,u,Lam,D,nEdd,M9,Inc,Z,Alpha):
    val = dBdT_lam(T,Lam/(1+Z))*Flam0(Lam/(1+Z),D,M9,Inc,Alpha)*u*(1+Z)
    return(val)

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

def timefunc(td,tp,tp_gap,R,Inc,Z):
    t0 = R/3e10/86400*np.sin(Inc*np.pi/180)*(1+Z)
    t1 = np.max([tp,td-t0]) ### t1,t2 in this code are t3,t4 in the paper
    t2 = np.min([tp+tp_gap,td+t0]) 
    t3 = np.max([tp-tp_gap,td-t0]) 
    t4 = np.min([tp,td+t0]) 
    val1 = G1(t1,t2,tp_gap,td,t0) + (tp+tp_gap-td)/tp_gap*G2(t1,t2,tp_gap,td,t0) 
    val2 = -1*G1(t3,t4,tp_gap,td,t0) + (td-tp+tp_gap)/tp_gap*G2(t3,t4,tp_gap,td,t0) #this version is right 
    val = val1+val2
    return(val)


