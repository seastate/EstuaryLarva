
import numpy as np


#line_start = lambda line, start: line[0:min(len(line),len(start))]==start
line_parse = lambda line, start: line[len(start)+1:-1] if line[0:min(len(line),len(start))]==start else False



class octaveStruct():
    """
    A class to facilitate input (and, maybe someday, output) of structures
    saved in GNU Octave.
    """
    def __init__(self,infile=None,verbose=False,readfile=True):
        """Create an octaveStruct instance, with an option to load parameters from a specified
           input file.
        """
        self.infile = infile
        self.verbose = verbose
        #
        self.ostr = {}
        #
        if readfile:
            self.readfile()

    def readfile(self,infile=None):
        if infile is not None:
            self.infile = infile
        print(f'Attempting to load octave structure from input file {self.infile}...')
        try:
            self.f = open(self.infile, "r")
        except IOError as e:
            print("I/O error(%s): %s %s" % (e.errno, e.strerror, self.infile))
            return
        # print header line
        header = self.f.readline()
        if self.verbose:
            print(f'Header line is:\n{header}')
        # read lines from the input file, breaking at EOF
        while True:
            line = self.f.readline()
            if self.verbose:
                print(line)
            if not line:
                print('Reached end of file (EOF)')
                break
            # check for name directive defining new variable
            new_name = line_parse(line,'# name:')
            if self.verbose:
                print(f'new_name = |{new_name}|')
            if new_name:
                # parse type of new variable
                line = self.f.readline()
                new_type = line_parse(line,'# type:')
                if self.verbose:
                    print(f'new_type = |{new_type}|')
                    print(f'new_name = |{new_name}|')
                if new_type == 'matrix':
                    self.parse_matrix(name=new_name)
                elif new_type == 'scalar':
                    self.parse_scalar(name=new_name)
                elif new_type == 'sq_string':
                    self.parse_string(name=new_name)
        # close input file
        self.f.close()
        #return dictionary of results
        return self.ostr

    def parse_scalar(self,name=None):
        """Parse a scalar and add it to the sys dictionary with key 'name'
        """
        if self.verbose:
            print(f'Parsing scalar {name} as float...')
        s = self.f.readline()
        s_float = float(s)
        self.ostr.update({name:s_float})
                
    def parse_string(self,name=None):
        """Parse a string and add it to the sys dictionary with key 'name'
        """
        if self.verbose:
            print(f'Parsing string {name} ...')
        # discard two comment lines, then record the string
        c1 = self.f.readline()
        c2 = self.f.readline()
        s = self.f.readline()
        self.ostr.update({name:s})
                
    def parse_matrix(self,name=None):
        """Parse a matrix and add it to the sys dictionary with key 'name'
        """
        if self.verbose:
            print(f'Parsing matrix {name} as float array...')
        line = self.f.readline()
        rows = int(line_parse(line,'# rows:'))
        if self.verbose:
            print(f'Matrix has {rows} rows')
        line = self.f.readline()
        cols = int(line_parse(line,'# columns:'))
        if self.verbose:
            print(f'Matrix has {cols} columns')
        A = np.zeros([rows,cols])
        for i in range(rows):
            # read line and parse, dropping the leading space
            sline = self.f.readline().split(' ')[1:]
            if self.verbose:
                print(sline)
            A[i,:] = [float(a) for a in sline]
        self.ostr.update({name:A})
                
    def keys(self):
        """Print keys of the ostr dictionary
        """
        print(f'Keys are {self.ostr.keys()}')

    def values(self,keys=[]):
        """Print values for all (default) or only listed keys
        """
        ostr_keys = list(self.ostr.keys())
        print(keys,len(keys))
        if len(keys) == 0: # display values for all keys
            keys = ostr_keys
        for key in keys:
            if key in ostr_keys:
                print(f'\n\nValue for key {key} is: \n{self.ostr[key]}')
            else:
                print(f'\n\nKey {key} not found......')


    def shapes(self,keys=[]):
        """Print shapes for all (default) or only listed keys, if they are arrays
        """
        ostr_keys = list(self.ostr.keys())
        print(keys,len(keys))
        if len(keys) == 0: # display values for all keys
            keys = ostr_keys
        for key in keys:
            if key in ostr_keys:
                try:
                    print(f'\n\nShape for key {key} is: \n{self.ostr[key]}.shape')
                except:
                    pass


                




