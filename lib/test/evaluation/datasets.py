from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    vtuav=DatasetInfo(module='lib.test.evaluation.VTUAV', class_name="VTUAVDataset", kwargs=dict(split='test')),
    vedai=DatasetInfo(module='lib.test.evaluation.VEDAI', class_name="VEDAIDataset", kwargs=dict(split='test')),
    dronevehicle=DatasetInfo(module='lib.test.evaluation.DroneVehicle', class_name="DroneVehicleDataset",
                             kwargs=dict(split='test')),
    dtest=DatasetInfo(module='lib.test.evaluation.DroneVehicle_test', class_name="DroneVehicleDataset",
                             kwargs=dict(split='test')),
    dronevehicle_norandom=DatasetInfo(module='lib.test.evaluation.DroneVehicle_norandom',
                                      class_name="DroneVehicleDataset", kwargs=dict(split='rtest')), # rtest or rtest_hom
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset