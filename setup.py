"""Setup"""

from setuptools import setup, find_packages

exclude = ['data', 'info', 'reports', 'scripts']

setup(name='lbgdl',
      version="0.0.1",
      description="Description",
      #   url='https://github.com/ese-msc-2022/irp-gpm22',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      author="Gustavo Pires Ferreira Machado",
      author_email='gustavo.machado04@yahoo.com.br',
      license='MIT',
      packages=find_packages(exclude=exclude)
      )
