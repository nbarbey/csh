#!/usr/bin/env python
"""
Command-line interface to perform PACS map-making with an
state-of-the-art PACS instrument model including various compression
model and sparse estimation.

XXX WORK IN PROGRESS

Author: Nicolas Barbey
"""

options = "hvf:o:"
long_options = ["help", "verbose", "filenames=", "output="]

compressions = {"averaging":csh.averaging,
                "cs":csh.cs}

def main():
    import os, sys, getopt, ConfigParser, time
    # parse command line arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], options, long_options)
    except getopt.GetoptError, err:
        print(str(err))
        usage()
        sys.exit(2)

    # default
    verbose = False

    # parse options
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        if o in ("-v", "--verbose"):
            verbose = True

    # read config file
    if len(args) == 0:
        print("Error: config filename is mandatory.\n")
        usage()
        sys.exit(2)
    config_file = args[0]
    config = ConfigParser.RawConfigParser()
    config.read(config_file)
    keywords = dict()
    # parse config
    for section in config.sections():
        keywords[section] = dict()
        for option in config.options(section):
            get = config.get
            # recast to bool, int or float if needed
            # if option not handled here, it defaults to a string
            if section == "PacsObservation":
                if option == "reject_bad_line":
                    get = config.getboolean
                if option == "fine_sampling_factor":
                    get = config.getint
                if option in ("active_fraction",
                              "delay",
                              "calblock_extension_time"):
                    get = config.getfloat
            if section == "scanline_masking":
                if option in ("n_repetition", "n_scanline"):
                    get = config.getint
            if section == "get_tod":
                if option in ("flatfielding",
                              "substraction_mean",
                              "raw"):
                    get = config.getboolean
            if section == "deglitching":
                if option in ("length", "nsigma"):
                    get = config.getfloat
            if section == "filter_median":
                if option == "length":
                    get = config.getfloat
            if section == "Projection":
                if option in ("oversampling",
                              "packed"):
                    get = config.getboolean
                if option == "npixels_per_sample":
                    get = config.getint
                if option == "resolution":
                    get = config.getfloat
            if section == "mapper_rls":
                if option in ("verbose",
                              "criterion"):
                    get = config.getboolean
                if option == "maxiter":
                    get = config.getint
                if option in ("hyper",
                              "tol"):
                    get = config.getfloat
            # store option using the appropriate get to recast option.
            keywords[section][option] = get(section, option)
    # special case for the main section
    data_file_list = config.get("main", "filenames").split(",")
    # if filenames argument is passed, override config file value.
    for o, a in opts:
        if o in ("-f", "--filenames"):
            data_file_list = a.split(", ")
    data_file_list = [w.lstrip().rstrip() for w in data_file_list]
    # append date string to the output file to distinguish results.
    date = time.strftime("%y%m%d_%H%M%S", time.gmtime())
    # extract filename from data_file
    filename = data_file_list[0].split(os.sep)[-1]
    # remove extension
    fname = ".".join(filename.split(".")[:-1])
    # store results into the Data subdirectory as expected by sumatra
    output_file = "Data/map" + fname + '_' + date + '.fits'
    # if output argument is passed, override config file value.
    for o, a in opts:
        if o in ("-o", "--output"):
            output_file = a
    # run tamasis mapper
    pipeline_huber(data_file_list, output_file, keywords, verbose=verbose)

def pipeline_huber(filenames, output_file, keywords, verbose=False):
    """
    Perform huber regularized inversion of state of the art
    PACS model.  The PACS model includes tod mask, compression,
    response, projection and noise covariance (invntt).
    Processing steps are as follows :
    
        - define PacsObservation instance
        - mask first scanlines to avoid drift
        - get Time Ordered Data (get_tod)
        - 2nd level deglitching (with standard projector and tod median
            filtering with a  narrow window)
        - median filtering
        - define projection
        - perform inversion on model
        - save file

    Arguments
    ---------
    filenames: list of strings
        List of data filenames.
    output_file : string
        Name of the output fits file.
    keywords: dict
        Dictionary containing options for all steps as dictionary.
    verbose: boolean (default False)
        Set verbosity.

    Returns
    -------
    Returns nothing. Save result as a fits file.
    """
    from scipy.sparse.linalg import cgs
    import tamasis as tm
    import linear_operators as lo
    # verbosity
    # define observation
    obs = tm.PacsObservation(filenames, **keywords["PacsObservation"])
    # extra masking
    tm.step_scanline_masking(obs, **keywords["scanline_masking"])
    # get data
    tod = obs.get_tod(**keywords["get_tod"])
    # deglitching
    # need to adapt degltiching to any compression model
    tm.step_deglitching(obs, tod, **keywords["deglitching"])
    # median filtering
    tod = tm.filter_median(tod, **keywords["filter_median"])
    # define projector
    projection = tm.Projection(obs, **keywords["Projection"])
    P = lo.aslinearoperator(projection)
    # build instrument model
    masking = tm.Masking(tod.mask)
    Mt = lo.aslinearoperator(masking)
    # compression
    mode = compressions[keywords["compression"].pop("mode")]
    factor = keywords["compression"].pop("factor")
    C = mode(data, factor, **keywords["compression"])
    # define map mask
    M = map_mask(tod, model, **keywords["map_mask"])
    # recast model as a new-style LinearOperator from linear_operators package
    H = Mt * C * P * M
    # vectorize data so it is accepted by LinearOperators
    y = tod.ravel()
    Ds = [tm.DiscreteDifference(axis=i, shapein=projection.shapein) for i in (0, 1)]
    Ds = [lo.aslinearoperator(D) for D in Ds]
    Ds = [D * M.T for D in Ds]
    # set tod masked values to zero
    tod = masking(tod)
    # perform map-making inversion
    hypers = (keywords["hacg"].pop("hyper"), ) * len(Ds)
    deltas = (keywords["hacg"].pop("delta"), ) * (len(Ds) + 1)
    map_huber = lo.hacg(H, tod.ravel(), priors=Ds, hypers=hypers,
                        deltas=deltas, **keywords["hacg"])
    # save
    map_huber = (M.T* map_huber).reshape(projection.shapein)
    map_huber = tm.Map(map_huber)
    map_huber.save(output_file)


def map_mask(tod, model, threshold=10):
    import linear_operators as lo
    import numpy as np

    return lo.decimate(model.T(np.ones(tod.shape)) < threshold)

def usage():
    print(__usage__)

__usage__ = """Usage: pacs_huber [options] [config_file]

Use tamasis map-making routine with csh compression and huber sparse
inversion.

[config_file] is the name of the configuration file which contains
map-making parameters. This file contains arguments to the various
processing steps of the map-maker. For an exemple, see the file
tamasis_invntt.cfg in the tamasis/pacs/src/ directory.

Options:
  -h, --help        Show this help message and exit.
  -v, --verbose     Print status messages to std output.
  -f, --filenames   Overrides filenames config file value.
  -o, --output      Overrides output default value.
"""

# to call from command line
if __name__ == "__main__":
    main()
