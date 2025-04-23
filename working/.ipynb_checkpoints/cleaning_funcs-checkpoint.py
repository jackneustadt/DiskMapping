import numpy as np
import glob,os
from astropy.io import ascii

c = 2.998e10

filt_dict = {'UVW2':2030, 'UVW1': 2589, 'UVM2': 2228, 'U': 3494, 'B': 4329, 'V': 5402,
            'u': 3590, 'g':4640, 'r': 6120, 'i':7440, 'z': 8897, 'BL': 4360, 'VL': 5460,
            '1180':1180, '1398':1398, '1502':1502, '1739':1739}

def chi2_offset(x1,x2,x3,y1,y2,y3,e1,e2,e3):
    num = (y1*(x3-x2)+y2*(x1-x3)+y3*(x2-x1))**2
    den = (e1**2)*((x3-x2)**2) + (e2**2)*((x1-x3)**2) + (e3**2)*((x1-x2)**2)
    time = (x1-x2)**2 + (x2-x3)**2 + (x1-x3)**2
    offset = (num-den)/time
    if offset < 0:
        return(0)
    else:
        return(np.sqrt(offset))

def collect_lc(agn):
    if agn == 'ngc5548':
        td_ij = []
        lam_ij = []
        lc_ij = np.array([])
        err_ij = []
        for file in glob.glob(base+'discs/phot/*.dat'):
            if 'phot' in file:
                filter_file = open(file, 'r')
                filter_lines = filter_file.readlines()
                filter_file.close()
                weff = round(float(filter_lines[0].split()[1])*1e4)
                temp_ij = []
                for line in filter_lines[1:]:
                    td_ij.append(float(line.split()[0]))
                    lam_ij.append(weff)
                    temp_ij.append(float(line.split()[1]))
                    err_ij.append(float(line.split()[2]))
                temp_ij = np.array(temp_ij)
                lc_ij = np.concatenate((lc_ij,temp_ij),axis=None)
        lc_ij = np.array(lc_ij)*1e16
        err_ij = np.array(err_ij)*1e16
        td_ij = np.array(td_ij)
        lam_ij = np.array(lam_ij)
        
        trim, = np.where(td_ij < 6900)
        lc_ij = lc_ij[trim]
        td_ij = td_ij[trim]
        lam_ij = lam_ij[trim]
        err_ij = err_ij[trim]
        
    if agn == 'mrk817':
        td_ij = np.array([])
        lam_ij = np.array([])
        lc_ij = np.array([])
        err_ij = np.array([])
        for file in glob.glob(base+'mrk817/phot/final/*.*'):
            table = ascii.read(file)  
            if 'cont' in file:
                filt = file.split('_')[1]
                if filt == 'V':
                    filt = 'VL'
                elif filt == 'B':
                    filt = 'BL'
                td_temp = table['col1']-2400000.5
                flam = table['col2']*100
                elam = table['col3']*100
                lam = filt_dict[filt]
                lam_temp = np.ones_like(flam)*lam
            else:
                filt = file.split('/')[-1].split('.')[0]
                lam = filt_dict[filt]
                td_temp = table['col1']-0.5
                flam = table['col2']*10
                elam = table['col3']*10
                lam_temp = np.ones_like(flam)*lam
            lc_ij = np.concatenate((lc_ij,flam),axis=None)
            err_ij = np.concatenate((err_ij,elam),axis=None)
            td_ij = np.concatenate((td_ij,td_temp),axis=None)
            lam_ij = np.concatenate((lam_ij,lam_temp),axis=None)
        lc_ij = np.array(lc_ij)
        err_ij = np.array(err_ij)
        td_ij = np.array(td_ij)
        lam_ij = np.array(lam_ij)
        
    return(lc_ij,err_ij,td_ij,lam_ij)

def apply_error_offset(lc_ij,err_ij,td_ij,lam_ij):
    lams = np.unique(lam_ij)
    for lam in lams:
        index, = np.where(lam == lam_ij)
    
        lc_lam = lc_ij[index]
        td_lam = td_ij[index]
        err_lam = err_ij[index]
    
        offset_lam = []
        
        prev = np.mean(err_lam/lc_lam)
    
        for i in range(1,len(index)-1):
            offset = chi2_offset(td_lam[i-1],td_lam[i],td_lam[i+1],
                                 lc_lam[i-1],lc_lam[i],lc_lam[i+1],
                                 err_lam[i-1],err_lam[i],err_lam[i+1])
            if td_lam[i+1]-td_lam[i-1] < 10:
                offset_lam.append(offset)
    
        err_add = np.median(offset_lam)
        err_ij[index] = np.sqrt((err_ij[index])**2+err_add**2)   
    return(lc_ij,err_ij,td_ij,lam_ij)

def export_clean(agn,path_to_clean,lc_ij,err_ij,td_ij,lam_ij):
    clean_file = path_to_clean+agn+'.dat'
    agn_array = np.array([list(td_ij),list(lam_ij.astype(int)),list(lc_ij),list(err_ij)])
    file = open(clean_file,'w')
    np.savetxt(file, agn_array.T)
    file.close()

def clean_data(agn,path_to_raw,path_to_clean,do_error_offset=True,do_export_clean=True):
    lc_ij,err_ij,td_ij,lam_ij = clean_lc(agn,path_to_raw)
    if do_error_offset:
        lc_ij,err_ij,td_ij,lam_ij = apply_error_offset(lc_ij,err_ij,td_ij,lam_ij)
    if do_export_clean:
        export_clean(agn,path_to_clean,lc_ij,err_ij,td_ij,lam_ij)
    return 
