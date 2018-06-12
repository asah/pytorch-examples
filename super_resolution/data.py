from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolder

def install_bsd300(dest=None):
    if dest:
        environ['QUILT_PRIMARY_PACKAGE_DIR'] = dest
    # force to avoid y/n prompt; does not re-download
    PKG = 'akarve/BSDS300'
    quilt.install(PKG, force=True)
    return quilt.load(PKG);

# use install_bsd300 instead
def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    pkg = install_bsd300()
    train = getattr(pkg, 'images', 'train')
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromPackage(train,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    pkg = install_bsd300()
    test = getattr(pkg, 'images', 'test')
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromPackage(test,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
