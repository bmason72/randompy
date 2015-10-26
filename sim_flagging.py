
#
# bsm - oct 2015 - simple demo power spectrum estimation &
#  effect of zeroing elements.
# 

import scipy as sp
import scipy.fftpack as spft
import pylab as pl

seed=72
sp.random.seed(seed)

nch=1000

# the lag ps will be only the zero or positive frequency parts-
if (sp.mod(nch,2) == 0):
    nlagch=nch/2+1
else:
    nlagch=(nch-1)/2+1

lag=sp.arange(nlagch)+0.1
# this is the square root of the lag PS, ie, the magnitude of the
#  coefficients
lagps=1.0/lag**0.5

freqs=spft.fftfreq(nch)

# convert into complex lag coefficients and
#  assign random phases before FFT'ing,
#  which will give us a realization of a 1/f lag power PS,
#  ie, one realization of a frequency spectrum.
# nb this includes the negative freq coefficients-
lagcoeff=sp.zeros(nch,dtype=complex)
lagcoeff[0]=lagps[0]
for i in range(nlagch):
    if (i > 0):        
        phase=sp.sqrt(-1.0) * sp.rand()
        #print phase,sp.exp(2.0*sp.pi*phase)
        lagcoeff[i] = lagps[i]*sp.exp(2.0*sp.pi*phase)
        lagcoeff[freqs == -1.0*freqs[i]] = lagps[i]*sp.exp(-2.0*sp.pi*phase)

spectrum=spft.fft(lagcoeff)

# first verify that the inverse FFT gives a consistent spectrum-
newlagps=spft.ifft(spectrum)
pl.plot(abs(freqs[0:nlagch]),abs(lagps))
pl.plot(abs(freqs[0:nlagch]),abs(newlagps[0:nlagch]),'r')
# nb - in the above, abs(freqs) is because for even # data points
#  the nyquist freq is given a negative freq.

flags=sp.ones(spectrum.size)
minflag=100
maxflag=250
flags[minflag:maxflag]=0
spectrum2=spectrum * flags

newlagps2=spft.ifft(spectrum2)
pl.plot(abs(freqs[0:nlagch]),abs(newlagps2[0:nlagch]),'g')
# the mean sqrt(power spectrum) is 13% or so low-
#sp.mean(abs(newlagps2[0:nlagch])/abs(lagps))
#sp.median(abs(newlagps2[0:nlagch])/abs(lagps))
#

spectrum3=spectrum*flags 
spectrum3[minflag:maxflag] = sp.median(spectrum) + sp.randn(maxflag-minflag) \
                             * sp.std(spectrum)

newlagps3=spft.ifft(spectrum3)
pl.plot(abs(freqs[0:nlagch]),abs(newlagps3[0:nlagch]),'k')
# this result is within a couple percent in sqrt(ps) unbiased
#sp.mean(abs(newlagps3[0:nlagch])/abs(lagps))
#sp.median(abs(newlagps3[0:nlagch])/abs(lagps))
