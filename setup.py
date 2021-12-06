from setuptools import find_packages, setup


requirements = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
]


extras_require = {
    'docs': ['sphinx', 'sphinx_rtd_theme', 'pygments-enaml'],
    'test': ['pytest', 'pytest-benchmark'],
}


setup(
    name='psiaudio',
    author='Brad Buran',
    author_email='info@bradburan.com',
    install_requires=requirements,
    extras_require=extras_require,
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE.txt',
    description='Audio tools',
    entry_points={
        'console_scripts': [],
    },
)
