import os
import csh

# define data set
datadir = os.getenv('CSH_DATA')
ids = ['1342184518', '1342184519', '1342184596', '1342184597', 
       '1342184598', '1342184599']
filenames = [os.path.join(datadir, id_str + '_blue_PreparedFrames.fits')
             for id_str in ids]
# no compression
output_path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',)
# compression modes
compression = ""
# median filter length
deglitch=True
covariance=True
decompression=True
filtering = True
filter_length = 10000
hypers = (1e9, 1e9)
ext = ".fits"
pre = "abell2218_madmap1_"
# to store results
# find a map for each compression and save it
sol = csh.rls(filenames, compression=compression, hypers=hypers, 
             deglitch=deglitch, covariance=covariance,
             decompression=decompression,
             filtering=filtering, filter_length=filter_length)
fname = os.path.join(output_path, pre + compression + ext)
sol.writefits(fname)
