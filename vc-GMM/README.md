
# Dependencies
* Cmake
* Blaze
* GCEM
* Homebrew - mac only

Compilation requires a C++14 compiler

# Installing Cmake

## On Mac

1. start by installing homebrew
~~~~
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
~~~~

2. proceed to the Cmake installation
~~~~
brew install cmake
~~~~

## On Linux
~~~~
sudo apt-get install cmake
~~~~

# Installing Premake 4

## On Mac

assuming homebrew was already installed (see previous section)
~~~~
brew install premake
~~~~

## On Linux
~~~~
sudo apt install premake4
~~~~

# Installing Blaze

1. get the source code from here: https://bitbucket.org/blaze-lib/blaze/src/master/
2. on linux install dependencies (nothing to do on mac)
~~~~
apt-get install libblas-dev liblapack-dev
~~~~

3. generate Makefiles and install
~~~~
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/
sudo make install
~~~~

# Installing GCEM

1. clone from github
~~~~
git clone https://github.com/kthohr/gcem ./gcem
~~~~

2. make a build directory:
~~~~
cd ./gcem
mkdir build
cd build
~~~~

3. generate Makefiles and install
~~~~
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/
sudo make install
~~~~

# Installing NPY C++ library to /usr/local/
~~~~
git clone https://github.com/rogersce/cnpy.git ./cnpy
mkdir build && cd build
cmake ./cnpy
make
make install
~~~~
