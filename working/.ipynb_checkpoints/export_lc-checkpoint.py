import numpy as np
import glob,os
from astropy.io import ascii
from import_lc import import_lc

c = 2.998e10

base = '/home/neustadt.7/data/'

agn_list = ['NGC 5548','Mrk 817','NGC 4151', 'Mrk 110', 'Mrk 142', 'Fairall 9', 'Mrk 509', 'NGC 4593']

for agn in agn_list:
    lc_ij,err_ij,td_ij,lam_ij = import_lc(agn)    
    if agn != 'Mrk 817':
        td_ij += 50000
    
    agn_file = agn.lower().replace(' ','')
    agn_file = base + 'mrk817/lcs/'+agn_file+'.dat'
    
    agn_array = np.array([list(td_ij),list(lam_ij.astype(int)),list(lc_ij),list(err_ij)])
    file = open(agn_file,'w')
    np.savetxt(file, agn_array.T)
    file.close()
    print('exporting '+agn)
    
#     if agn == 'Mrk 110':
#         file = open(agn_file,'r')
#         for line in file:
#             print(line[0])

    
print('export complete')