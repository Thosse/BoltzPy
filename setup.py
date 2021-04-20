from distutils.core import setup
# from Cython.Build import cythonize
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

# This is outsourced from setup for build_sphinx
name = 'BoltzPy'
version = '0.2'
release = version + '.0'
author = 'Thomas Sasse'

setup(
    name=name,
    version=release,
    description='Deterministic Solver for the Boltzmann Equation',
    author=author,
    author_email='thomas.sasse@tu-ilmenau.de',
    url='https://github.com/Thosse/BoltzPy',
    license='Apache Software License',	
    packages=['boltzpy'],  
    cmdclass=cmdclass,
    # these are optional and override conf.py settings
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'copyright': ('setup.py', '2021, ' + author),
            'fresh_env': ('', True),   
            'source_dir': ('', 'doc/source/'),
            'build_dir': ('', 'doc/build/'),
            'config_dir': ('','doc/source/')
            }}
)
