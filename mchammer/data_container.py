import getpass
import json
import numpy as np
import pandas as pd
import socket
import tarfile
import tempfile

from ase import Atoms
from ase.io import write as ase_write, read as ase_read
from collections import OrderedDict
from datetime import datetime
from icet import __version__ as icet_version
from typing import BinaryIO, Dict, List, TextIO, Tuple, Union


class DataContainer:
    """
    Data container for storing information concerned with
    Monte Carlo simulations performed with mchammer.

    Parameters
    ----------
    atoms : ASE Atoms object
        reference atomic structure associated with the data container

    name_ensemble : str
        name of associated ensemble

    random_seed : int
        seed used in random number generator
    """

    def __init__(self, atoms: Atoms, ensemble_name: str, random_seed: int):
        """
        Initializes a DataContainer object.
        """

        if not isinstance(atoms, Atoms):
            raise TypeError('atoms is not an ASE Atoms object')

        self.atoms = atoms.copy()

        self._observables = []
        self._parameters = OrderedDict()
        self._metadata = OrderedDict()
        self._data = pd.DataFrame(columns=['mctrial'])
        self._data = self._data.astype({'mctrial': int})

        self.add_parameter('seed', random_seed)

        self._metadata['ensemble_name'] = ensemble_name
        self._metadata['date_created'] = \
            datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        self._metadata['username'] = getpass.getuser()
        self._metadata['hostname'] = socket.gethostname()
        self._metadata['icet_version'] = icet_version

    def add_observable(self, tag: str):
        """
        Adds observable name.

        Parameters
        ----------
        tag
            name of observable

        Raises
        ------
        TypeError
            if input parameter has the wrong type
        """
        if not isinstance(tag, str):
            raise TypeError(f'tag has the wrong type: {type(tag)}')
        if tag not in self._observables:
            self._observables.append(tag)

    def add_parameter(self, tag: str,
                      value: Union[int, float, List[int], List[float]]):
        """
        Adds parameter associated with underlying ensemble.

        Parameters
        ----------
        tag
            parameter name
        value
            parameter value

        Raises
        ------
        TypeError
            if input parameters have the wrong type
        """
        import copy
        if not isinstance(tag, str):
            raise TypeError(f'tag has the wrong type: {type(tag)}')
        if not isinstance(value, (int, float, list)):
            raise TypeError(f'value has the wrong type: {type(value)}')
        self._parameters[tag] = copy.deepcopy(value)

    def append(self, mctrial: int,
               record: Dict[str, Union[int, float, list]]):
        """
        Appends data to data container.

        Parameters
        ----------
        mctrial
            current Monte Carlo trial step
        record
            dictionary of tag-value pairs representing observations

        Raises
        ------
        TypeError
            if input parameters have the wrong type

        Todo
        ----
        * This might be a quite expensive way to add data to the data
          frame. Testing and profiling to be carried out later.
        """
        if not isinstance(mctrial, int):
            raise TypeError(f'mctrial has the wrong type: {type(mctrial)}')
        if not isinstance(record, dict):
            raise TypeError(f'record has the wrong type: {type(record)}')
        row_data = OrderedDict()
        row_data['mctrial'] = mctrial
        row_data.update(record)
        self._data = self._data.append(row_data, ignore_index=True)

    def get_data(self, tags: List[str]=None,
                 start: int=None, stop: int=None, interval: int=1,
                 fill_method: str=None) -> Union[list, Tuple[list, list]]:
        """Returns the accumulated data for the requested observables.

        Parameters
        ----------
        tags
            tags of the requested properties; by default all columns
            of the data frame will be returned in lexicographical
            order.

        start
            minimum value of trial step to consider; by default the
            smallest value in the mctrial column will be used.

        stop
            maximum value of trial step to consider; by default the
            largesst value in the mctrial column will be used.

        interval
            increment for mctrial; by default the smallest available
            interval will be used.

        fill_method : {'skip_none', 'fill_backward', 'fill_forward',
                       'linear_interpolate', None}
            method employed for dealing with missing values

        Raises
        ------
        ValueError
            if observables are requested that are not in data container
        ValueError
            if fill method is unknown
        """
        fill_methods = ['skip_none',
                        'fill_backward',
                        'fill_forward',
                        'linear_interpolate']

        if tags is None:
            tags = self._data.columns.tolist()
        else:
            for tag in tags:
                if tag not in self._data:
                    raise ValueError(f'No observable named {tag} in data'
                                     ' container')

        if start is None and stop is None:
            data = self._data.loc[::interval, tags]
        else:
            data = self._data.set_index(self._data.mctrial)

            if start is None:
                data = data.loc[:stop:interval, tags]
            elif stop is None:
                data = data.loc[start::interval, tags]
            else:
                data = data.loc[start:stop:interval, tags]

        if fill_method is not None:
            if fill_method not in fill_methods:
                raise ValueError(f'Unknown fill method: {fill_method}')

            # retrieve only valid observations
            if fill_method is 'skip_none':
                data.dropna(inplace=True)

            else:
                # fill NaN with the next valid observation
                if fill_method is 'fill_backward':
                    data.fillna(method='bfill', inplace=True)
                # fill NaN with the last valid observation
                elif fill_method is 'fill_forward':
                    data.fillna(method='ffill', inplace=True)
                # fill NaN with the linear interpolation
                # of the last and next valid observations
                elif fill_method is 'linear_interpolate':
                    data.interpolate(limit_area='inside', inplace=True)

                # drop any left-over nan value
                data.dropna(inplace=True)

        data_list = []
        for tag in tags:
            data_list.append(
                # convert NaN to None
                [None if np.isnan(x).any() else x for x in data[tag]])
        if len(tags) > 1:
            # return a tuple if more than one tag is given
            return tuple(data_list)
        else:
            # return a list if only one tag is given
            return data_list[0]

    @property
    def data(self) -> pd.DataFrame:
        """ pandas data frame (see :class:`pandas.DataFrame`) """
        return self._data

    @property
    def parameters(self) -> dict:
        """ parameters associated with Monte Carlo simulation """
        return self._parameters.copy()

    @property
    def observables(self) -> List[str]:
        """ observable names """
        return self._observables

    @property
    def metadata(self) -> dict:
        """ metadata associated with data container """
        return self._metadata

    def reset(self):
        """ Resets (clears) data frame of data container. """
        self._data = pd.DataFrame()

    def get_number_of_entries(self, tag: str=None) -> int:
        """
        Returns the total number of entries with the given observable tag.

        Parameters
        ----------
        tag
            name of observable; by default the total number of rows in the
            data frame will be returned.

        Raises
        ------
        ValueError
            if observable is requested that is not in data container
        """
        if tag is None:
            return len(self._data)
        else:
            if tag not in self._data:
                raise ValueError(f'No observable named {tag}'
                                 ' in data container')
            return self._data[tag].count()

    def get_average(self, tag: str,
                    start: int=None, stop: int=None) -> Tuple[float, float]:
        """
        Returns average and standard deviation of a scalar observable.

        Parameters
        ----------
        tag
            tag of field over which to average
        start
            minimum value of trial step to consider. If None, lowest value
            in the mctrial column will be used.
        stop
            maximum value of trial step to consider. If None, highest value
            in the mctrial column will be used.

        Raises
        ------
        ValueError
            if observable is requested that is not in data container
        TypeError
            if requested observable is not of a scalar data type
        """
        if tag not in self._data:
            raise ValueError(f'No observable named {tag} in data container')

        if self._data[tag].dtype not in ['int64', 'float64']:
            raise TypeError(f'Data for {tag} is not scalar')

        if start is None and stop is None:
            return self._data[tag].mean(), self._data[tag].std()
        else:
            data = self.get_data(tags=[tag], start=start, stop=stop,
                                 fill_method='skip_none')
            return np.mean(data), np.std(data)

    @staticmethod
    def read(infile: Union[str, BinaryIO, TextIO]):
        """
        Reads DataContainer object from file.

        Parameters
        ----------
        infile
            file from which to read

        Raises
        ------
        FileNotFoundError
            if file is not found (str)
        ValueError
            if file is of incorrect type (not a tarball)
        """
        import os

        if isinstance(infile, str):
            filename = infile
            if not os.path.isfile(filename):
                raise FileNotFoundError
        else:
            filename = infile.name

        if not tarfile.is_tarfile(filename):
            raise ValueError(f'{filename} is not a tar file')

        reference_atoms_file = tempfile.NamedTemporaryFile()
        reference_data_file = tempfile.NamedTemporaryFile()
        runtime_data_file = tempfile.NamedTemporaryFile()

        with tarfile.open(mode='r', name=filename) as tar_file:
            # file with atoms
            reference_atoms_file.write(tar_file.extractfile('atoms').read())

            reference_atoms_file.seek(0)
            atoms = ase_read(reference_atoms_file.name, format='json')

            # file with reference data
            reference_data_file.write(
                tar_file.extractfile('reference_data').read())
            reference_data_file.seek(0)
            reference_data = json.load(reference_data_file)

            # init DataContainer
            dc = DataContainer(atoms,
                               reference_data['metadata']['ensemble_name'],
                               reference_data['parameters']['seed'])
            for key in reference_data:
                if key == 'metadata':
                    for tag, value in reference_data[key].items():
                        if tag == 'ensemble_name':
                            continue
                        dc._metadata[tag] = value
                elif key == 'parameters':
                    for tag, value in reference_data[key].items():
                        if tag == 'seed':
                            continue
                        dc.add_parameter(tag, value)
                elif key == 'observables':
                    for value in reference_data[key]:
                        dc.add_observable(value)

            # add runtime data from file
            runtime_data_file.write(
                tar_file.extractfile('runtime_data').read())

            runtime_data_file.seek(0)
            runtime_data = pd.read_json(runtime_data_file)
            dc._data = runtime_data.sort_index(ascending=True)

        return dc

    def write(self, outfile: Union[str, BinaryIO, TextIO]):
        """
        Writes DataContainer object to file.

        Parameters
        ----------
        outfile
            file to which to write
        """
        self._metadata['date_last_backup'] = \
            datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        # Save reference atomic structure
        reference_atoms_file = tempfile.NamedTemporaryFile()
        ase_write(reference_atoms_file.name, self.atoms, format='json')

        # Save reference data
        reference_data = {'observables': self._observables,
                          'parameters': self._parameters,
                          'metadata': self._metadata}

        reference_data_file = tempfile.NamedTemporaryFile()
        with open(reference_data_file.name, 'w') as handle:
            json.dump(reference_data, handle)

        # Save pandas DataFrame
        runtime_data_file = tempfile.NamedTemporaryFile()
        self._data.to_json(runtime_data_file.name, double_precision=15)

        with tarfile.open(outfile, mode='w') as handle:
            handle.add(reference_atoms_file.name, arcname='atoms')
            handle.add(reference_data_file.name, arcname='reference_data')
            handle.add(runtime_data_file.name, arcname='runtime_data')
        runtime_data_file.close()
