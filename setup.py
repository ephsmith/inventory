from setuptools import setup

setup(name='inventory',
      version='1.0',
      description='Python based inventory system for a PLC based pick-n-place',
      author='Forrest Smith',
      license='MIT',
      packages=['plc', 'inventory'],
      zip_save=False)
