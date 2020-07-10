from setuptools import setup, find_packages
import receipt2vec


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='receipt2vec',
    url='https://github.com/TinkoffCreditSystems/receipt2vec',
    license='Apache License 2.0',
    version=receipt2vec.__version__,
    author='Alexey Pichugin',
    author_email="a.o.pichugin@tinkoff.ru",
    description='Tool for feature extracting from receipts from Russian stores',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Natural Language :: Russian',
    ],
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={
        '': ['data/*.model', 'models/models_data/*.tar', 'requirements.txt'],
    },
    dependency_links=['torch_data_utils @ git+git://github.com/Nemexur/torch_data_utils'],
    install_requires=required,
    entry_points={
        'console_scripts': ['receipt2vec = receipt2vec.__main__:main']
    },
    zip_safe=False,
)
