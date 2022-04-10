#solving the TOV equation: works in CGS units
#input EOS file = eos.d, with columns = eps prs nB in nuclear units 
# Copyright (C) 2022  Ankit Kumar
# Email: akvyas1995@gmail.com
#-----------------------------------------------------------------------
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#-----------------------------------------------------------------------


import numpy as np
from math import pi
from scipy.interpolate import interp1d


global c,G,msol,edcf

edcf = 1.602176565e33               #energy density from MeV/fm^3 to ergs/cm^3

c = 2.99792458e10
G = 6.6743e-8
msol = 1.98847e33

#importing the EOS from eos.d
eos = np.genfromtxt('eos.d')
eps,prs,nB = edcf*eos[:,0],edcf*eos[:,1],eos[:,2]
eps_int = interp1d(prs,eps,kind='cubic',bounds_error=False,fill_value='extrapolate')
prs_int = interp1d(nB,prs,kind='cubic',bounds_error=False,fill_value='extrapolate')


#RHS of pressure equation
def diff_p(r,e,p,m):
    return -G/(r*c**2)*(e+p)*(m*c**2+4*pi*r**3*p)/(r*c**2-2*G*m)
    
#RHS of mass equation
def diff_m(r,e):
    return 4*pi*r**2*e/c**2


R_limit = 25e5
dr = 0.02e5

#number of stars in the sequence
npts_seq = 51

#central density values to iterate over
nB0_min,nB0_max = 0.2,np.max(nB)
nB0_step = (nB0_max-nB0_min)/(npts_seq-1)
nB0 = np.arange(nB0_min,nB0_max+nB0_step,nB0_step)

M,R = np.zeros(npts_seq,float),np.zeros(npts_seq,float)
for i_seq in range(npts_seq):
    
    r0 = dr
    p0 = prs_int(nB0[i_seq])
    m0 = dr*diff_m(dr,eps_int(p0))

    #4th order Runge-Kutta routine
    while True:

        k1p = dr*diff_p(r0,eps_int(p0),p0,m0)
        k1m = dr*diff_m(r0,eps_int(p0))

        k2p = dr*diff_p(r0+dr/2,eps_int(p0+k1p/2),p0+k1p/2,m0+k1m/2)
        k2m = dr*diff_m(r0+dr/2,eps_int(p0+k1p/2))
        
        k3p = dr*diff_p(r0+dr/2,eps_int(p0+k2p/2),p0+k2p/2,m0+k2m/2)
        k3m = dr*diff_m(r0+dr/2,eps_int(p0+k2p/2))
        
        k4p = dr*diff_p(r0+dr,eps_int(p0+k3p),p0+k3p,m0+k3m)
        k4m = dr*diff_m(r0+dr,eps_int(p0+k3p))

        r = r0 + dr
        p = p0 + (k1p+2*k2p+2*k3p+k4p)/6
        m = m0 + (k1m+2*k2m+2*k3m+k4m)/6
        
        if p<0:
            break

        r0,p0,m0 = r,p,m
    
    M[i_seq] = (m0+m)/2/msol
    R[i_seq] = (r0+r)/2/1e5


#position of maximum mass
M = np.array(M,float)
R = np.array(R,float)

#exporting the data to a file seq_tov.d
np.savetxt(f'seq_tov.d',np.stack((M,R,nB0),axis=1),fmt="%f",header='M(msol) \t R(km) cent_nB(fm^-3)')