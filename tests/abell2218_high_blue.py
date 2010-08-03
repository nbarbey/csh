import os
import csh

# define data set
datadir = os.getenv('CSH_DATA')
filenames = [datadir + '/1342184598_blue_PreparedFrames.fits[5954:67614]',
             datadir + '/1342184599_blue_PreparedFrames.fits[5954:67615]']
# no compression
output_path = os.path.join(os.getenv('HOME'), 'data', 'csh', 'output',)
# compression modes
compressions = ["", "ca", "cs"]
#compressions = ["ca"]
# median filter length
deglitch=True
covariance=False
filtering = True
filter_length = 100
ext = ".fits"
pre = "abell2218_high_blue_rls_"
# to store results
sol = []
# define same header for all maps
tod, projection, header, obs = csh.load_data(filenames)
# get the weight map
weights = projection.transpose(tod.ones(tod.shape))
weights.writefits(os.path.join(output_path, pre + 'weights' + ext))
del tod, projection, obs
# find a map for each compression and save it
for comp in compressions:
    if comp == "":
        hypers = (1/8., 1/8.)
    else:
        hypers = (1e0, 1e0)
    sol.append(csh.rls(filenames, compression=comp, hypers=hypers, 
                       header=header,
                       deglitch=deglitch, covariance=covariance,
                       filtering=filtering, filter_length=filter_length))
    fname = os.path.join(output_path, pre + comp + ext)
    sol[-1].writefits(fname)
