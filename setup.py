from setuptools import setup

setup(name='torchaudio_contrib',
      version='0.1',
      description='To propose audio processing Pytorch codes with nice and easy-to-use APIs and functionality',
      url='https://github.com/keunwoochoi/torchaudio-contrib',
      author='Keunwoo Choi, Faro Stöter, Kiran Sanjeevan, Jan Schlüter',
      author_email='gnuchoi@gmail.com',
      license='MIT',
      install_requires=['torch'],
      packages=['torchaudio_contrib'],
      zip_safe=False)
