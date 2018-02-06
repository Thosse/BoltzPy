from distutils.core import setup
# from Cython.Build import cythonize
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

name = 'BoltzPy'
version = '0.1'
release = '0.1'

setup(
    name=name,
    author='Thomas Sasse',
    version=release,
    cmdclass=cmdclass,
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'copyright': ('setup.py', '2017, ' + name)
            }}#,
#    ext_modules = cythonize("spt/cyth.pyx")
)
