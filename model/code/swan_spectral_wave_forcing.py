import xarray as xr
import numpy as np

# Write this spectrum to a swan spectrum file for forcing. The function below was found at this link 
# (https://github.com/caiostringari/swantools/blob/master/swantools/io/SwanIO.py) and has been modfied.
#  by EJ Rainville and Malcolm LeClair

def write_stationary_spec2d(fname,freqs,dirs,spc,factor=10**-6):
	with open(fname,"w") as f:
		f.write("SWAN   1          Swan standard spectral file, version \n")
		f.write("AFREQ             absolute frequencies in Hz \n")
		f.write("     {}           number of frequencies \n".format(len(freqs)))
		for freq in freqs:
			f.write(" {} \n".format(str(np.round(freq, 4)).ljust(6,"0")))
		f.write("NDIR              spectral nautical directions in degr \n")
		f.write("    {}            number of directions \n".format(len(dirs)))
		for dir in dirs:
			f.write(" {}\n".format(str(np.round(dir, 4)).ljust(8,"0")))
		f.write("QUANT \n")
		f.write("     1            number of quantities in table \n")
		f.write("VaDens            variance densities in m2/Hz/degr \n")
		f.write("m2/Hz/degr        unit \n")
		f.write("   -0.9900E+02    exception value \n")
		f.write("FACTOR \n")
		f.write(" {} \n   ".format(factor))
		np.savetxt(f,spc*(1/factor), '%5.0f')