from setuptools import setup

classifiers = [
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Topic :: Scientific/Engineering'
]

setup(
    name='universal-divergence',
    version='0.2.0',
    author='Kazuaki Tanida',
    url='https://github.com/slaypni/universal-divergence',
    description='A divergence estimator of two sets of samples.',
    license='MIT',
    keywords=['KL', 'Kullback-Leibler', 'divergence', 'information measure'],
    package_dir={'universal_divergence': 'src'},
    packages=['universal_divergence'],
    install_requires=['numpy', 'scipy', 'joblib'],
    classifiers=classifiers
)
