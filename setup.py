import os.path

import setuptools

root_dir = os.path.abspath(os.path.dirname(__file__))


install_requires = [
    'loguru',
    'numpy',
    'ortools',
]

setuptools.setup(
    name='solution',
    version='1.0.0',
    description='An implementation of Contest solution',
    author='Zhen Wang',
    author_email='wangzhen.1105@foxmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    packages=['src', 'tests'],
    install_requires=install_requires,
)
