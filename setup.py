from setuptools import setup

setup(name='modular_rl',
      version='0.0.1',
      description='Implementation of TRPO and related algorithms',
      url='https://github.com/joschu/modular_rl',
      author='John Schulman',
      author_email='joschu@eecs.berkeley.edu',
      license='MIT',
      packages=['modular_rl'],
      install_requires=[
          'keras==1.0.1',
          'theano==0.8.2',
          'tabulate',
      ],
      zip_safe=False)
