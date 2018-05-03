import io
import sys
import os
import subprocess
from distutils.sysconfig import get_python_lib
from setuptools import setup
from setuptools.command.install import install

def get_virtualenv_path():
    """Used to work out path to install compiled binaries to."""
    if hasattr(sys, 'real_prefix'):
        return sys.prefix

    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        return sys.prefix

    if 'conda' in sys.prefix:
        return sys.prefix

    return None

class CustomInstall(install):
    """Custom handler for the 'install' command."""
    def run(self):
        self.compile_and_install_software()
        install.run(self)

    def compile_and_install_software(self):
        """Used the subprocess module to compile/install the C software."""
        src_path = './ndreg/'

        # compile the software
        cmd = 'cmake'
        venv = get_virtualenv_path()
#        if venv:
#            cmd += ' --DCMAKE_RUNTIME_OUTPUT_DIRECTORY={} .'.format(os.path.abspath(venv))
        print('Running command: {} in directory: {}'.format(cmd, os.path.abspath(src_path)))
        subprocess.check_call(cmd + ' .', cwd=src_path, shell=True)

        # install the software (into the virtualenv bin dir if present)
        subprocess.check_call('make', cwd=src_path, shell=True)
        subprocess.check_call('cp metamorphosis {}/bin/'.format(venv), cwd=src_path, shell=True)



# Package meta-data.
NAME = 'ndreg'
DESCRIPTION = 'Registration package that does affine and LDDMM registration'
URL = 'https://github.com/neurodata/ndreg'
EMAIL = 'vikramc@jhmi.edu'
AUTHOR = 'Vikram Chandrashekhar'

# What packages are required for this module to be executed?
REQUIRED = [
        'numpy', 'SimpleITK', 'scikit-image', 'tifffile'
]

#CMAKE_INSTALL_DIR = get_virtualenv_path() or ''
# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    #cmake_source_dir='./ndreg',
    #cmake_args=['--DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}'.format(CMAKE_INSTALL_DIR)],
#    cmake_install_dir=[],
    packages=['ndreg'],
    #package_data = {'ndreg': 'metamorphosis'},
    #scripts=['bin/*'],
    include_package_data=True,
    # If your package is a single module, use this instead of 'packages':
    #py_modules=['ndreg'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    #include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # add extension module
    # add custom build_ext command
#    ext_modules=[
#        Executable(
#            name='',
#            sources=['source.c', 'extra.cpp'],
#            libraries=['libsaveforlater']
#            language='c++',
#            include_dirs=['../include'],
#            extra_compile_args=['-static'],
#            extra_link_args=['-static']
#        )],
    zip_safe=False,
    cmdclass={
#        'build_ext': build_ext,
#        'upload': UploadCommand,
        'install': CustomInstall,
    },
)

