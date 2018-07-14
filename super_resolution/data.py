from operator import attrgetter
from os.path import exists, join, basename
from os import makedirs, remove
from six import string_types
from six.moves import urllib
import tarfile

from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from quilt.asa.pytorch import dataset
from quilt.nodes import DataNode
from PIL import Image
import quilt

def install_data():
    # force to avoid y/n prompt; does not re-download
    PKG = 'akarve/BSDS300'
    quilt.install(PKG, force=True)

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def node_parser(node):
    path = node()
    if isinstance(path, string_types):
        img = Image.open(path).convert('YCbCr')
        y, _, _ = img.split()
        return y
    else:
        raise TypeError('Expected string path to an image fragment')


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    def _inner(img):
        img_ = img.copy()
        return Compose([
            CenterCrop(crop_size),
            ToTensor(),
        ])(img_)
    
    return _inner

def is_image(node):
    """file extension introspection on Quilt nodes"""
    if isinstance(node, DataNode):
        filepath = node._meta.get('_system', {}).get('filepath')
        if filepath:
            return any(
                filepath.endswith(extension)
                for extension in [".png", ".jpg", ".jpeg"])

def get_training_set(upscale_factor):
    install_data()
    from quilt.data.akarve import BSDS300 as bsds
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return bsds.images.train(
        asa=dataset(
            include=is_image,
            node_parser=node_parser,
            input_transform=input_transform(crop_size, upscale_factor),
            target_transform=target_transform(crop_size)
        )
    )
     
def get_test_set(upscale_factor):
    install_data()
    from quilt.data.akarve import BSDS300 as bsds
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return bsds.images.test(
        asa=dataset(
            include=is_image,
            node_parser=node_parser,
            input_transform=input_transform(crop_size, upscale_factor),
            target_transform=target_transform(crop_size)
        ) 
    )
 
