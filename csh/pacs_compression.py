#!/usr/bin/env python

import numpy as np

import tamasis as tm
import fitsarray as fa
import linear_operators as lo
import csh

options = "hvf:o:"
long_options = ["help", "verbose", "filenames=", "output="]

compressions = {"averaging":csh.averaging,
                "cs":csh.cs}

def main():
    """Parse config file and execute data compression."""
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
            if section == "get_tod":
                if option in ("flatfielding",
                              "substraction_mean",
                              "raw"):
                    get = config.getboolean
            # store option using the appropriate get to recast option.
            keywords[section][option] = get(section, option)
    # special case for the main section
    data_file_list = config.get("main", "filenames").split(", ")
    # if filenames argument is passed, override config file value.
    for o, a in opts:
        if o in ("-f", "--filenames"):
            data_file_list = a.split(", ")
    data_file_list = [w.rstrip().lstrip() for w in data_file_list]
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
    # check existence of output path
    outdir = os.path.dirname(output_file)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # run compression code
    compressed_data = generate_compressed_data(data_file_list, **keywords)
    # save result
    compressed_data.tofits(output_file)

def generate_compressed_data(filenames, **keywords):
    """
    Compress data to a given factor using a custom compression matrix
    or operator.
    """
    obs = tm.PacsObservation(filenames, **keywords["PacsObservation"])
    data = obs.get_tod(**keywords["get_tod"])
    # model
    A = tm.PacsConversionAdu(obs, **keywords["PacsConversionAdu"])
    mode = compressions[keywords["compression"].pop("mode")]
    factor = keywords["compression"].pop("mode")
    C = mode(data, factor, **keywords["compression"])
    # convert
    digital_data = A(data)
    # apply compression
    y = C * digital_data.ravel()
    # reshape and recast
    cshape = list(digital_data.shape)
    cshape[1] = np.ceil(cshape[1] / factor)
    # XXX discard extra pixels in cs
    if mode == csh.cs:
        cshape = list(data.shape)
        cshape[1] = np.floor(cshape[1] / factor)
        y = y[:np.prod(cshape)]
    compressed_data =  fa.FitsArray(data=y.reshape(cshape))
    return compressed_data

def usage():
    print(__usage__)

__usage__ = """Usage: pacs_compression [options] [config_file]

Use various compression scheme to compress raw PACS data.

[config_file] is the name of the configuration file.

Options:
  -h, --help        Show this help message and exit.
  -v, --verbose     Print status messages to standard output.
  -f, --filenames   Overrides filenames configuration file value.
  -o, --output      Overrides output default value.
"""

# to call from command line
if __name__ == "__main__":
    main()
