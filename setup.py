from pathlib import Path
from setuptools import find_packages, setup


requirements = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
]


extras_require = {
    'examples': ['sounddevice'],
    'docs': ['sphinx', 'sphinx_gallery', 'sphinx_rtd_theme', 'pygments-enaml',
             'sounddevice'],
    'test': ['pytest', 'pytest-benchmark', 'pytest-xdist'],
}


# Get version number
version_file = Path(__file__).parent / 'psiaudio' / '__init__.py'
for line in version_file.open():
    if line.strip().startswith('__version__'):
        version = line.split('=')[1].strip().strip('\'')
        break
else:
    raise RuntimeError('Could not determine version')


setup(
    name='psiaudio',
    author='psiaudio development team',
    install_requires=requirements,
    extras_require=extras_require,
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE.txt',
    description='Audio tools supporting psiexperiment',
    entry_points={
        'console_scripts': [],
    },
    version=version,
)
