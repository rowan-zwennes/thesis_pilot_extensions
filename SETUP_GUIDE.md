# Virtual Linux Computer Set-up Guide

This guide provides the necessary steps to configure a Debian-based Linux environment for replicating the computational experiments in this study. The primary steps involve setting up a virtual machine, installing system-level dependencies, configuring a specific Python version using `pyenv`, and compiling the required C++ libraries before building the final PILOT project.

To begin, you will need a virtual machine running a recent version of Debian or a Debian-derivative like Ubuntu. Free virtualization software such as **Oracle VirtualBox** or **VMware Workstation Player** can be used. Download the official installation ISO from the Debian or Ubuntu website and follow the standard installation procedure to create a fresh virtual machine.

Once the virtual machine is running and you have access to the terminal, execute the following commands. The steps are organized into logical blocks for clarity.

## 1. System Preparation and Essential Dependencies

First, update the system's package list and install all essential build tools and libraries required for compiling Python and other software components.

```bash
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev libgdbm-dev libnss3-dev \
    xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev \
    liblzma-dev software-properties-common ca-certificates
```

## 2. Python Environment Setup via `pyenv`

We use `pyenv` to manage Python versions, ensuring the exact version (3.10.13) used in this study is installed.

### a) Install `pyenv`

```bash
curl https://pyenv.run | bash
```

### b) Configure your shell to load `pyenv`

Add the following lines to the end of your `~/.bashrc` file. You can do this with a text editor like `nano ~/.bashrc`.

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

### c) Apply the changes and install Python

Close and reopen your terminal, or run `source ~/.bashrc`. Then, install Python 3.10.13.

```bash
pyenv install 3.10.13
```

## 3. Clone and Configure the PILOT Project

Clone this repository, set the local Python version, and create a dedicated virtual environment.

```bash
# Clone the project repository from your GitHub
git clone https://github.com/[your-github-username]/[your-repository-name].git

# Navigate into the newly created project directory
cd thesis_pilot_extensions/

# Set the Python version for this project directory
pyenv local 3.10.13

# Create and activate a Python virtual environment
python -m venv .myprojectenv
source .myprojectenv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
```

## 4. Compile and Install C++ Dependencies

The PILOT project depends on the C++ libraries Armadillo and CARMA. They must be compiled from source.

### a) Install Armadillo

```bash
# Install additional dependencies for C++ libraries
sudo apt-get install cmake g++ libopenblas-dev liblapack-dev

# Download and extract Armadillo
wget http://sourceforge.net/projects/arma/files/armadillo-14.0.3.tar.xz
tar -xvf armadillo-14.0.3.tar.xz
cd armadillo-14.0.3/

# Build and install
mkdir build && cd build
cmake ..
make
sudo make install
cd ../.. # Return to the PILOT project root
```

### b) Install CARMA

```bash
# Install Pybind11, a dependency for CARMA
pip install pybind11

# Clone, build, and install CARMA
git clone https://github.com/RUrlus/carma
cd carma/
mkdir build && cd build
cmake -DCARMA_INSTALL_LIB=ON ..
sudo cmake --build . --config Release --target install
cd ../.. # Return to the PILOT project root
```

## 5. Build the PILOT Project

With all dependencies in place, you can now compile the PILOT C++ extensions.

```bash
# Navigate into a new build directory within the PILOT project
mkdir build && cd build
cmake ..
make
```

## Troubleshooting: Python/NumPy Path Errors

In some environments, particularly if multiple Python versions are present, the `cmake ..` command may fail to locate the necessary Python or NumPy header files. This often results in an error message similar to this:

```
-- Could NOT find Python3 (missing: Python3_NumPy_INCLUDE_DIRS NumPy)
```

If this occurs, it can be resolved by manually specifying the paths. Open the `CMakeLists.txt` file located in the root directory of the PILOT project (e.g., with `nano ../CMakeLists.txt`). Add the following two lines to the top of the file:

```cmake
set(Python3_INCLUDE_DIR <path_to_python_include>)
set(Python3_NumPy_INCLUDE_DIR <path_to_numpy_include>)
```

Place them right below these lines:

```cmake
cmake_minimum_required(VERSION 3.12)
project(PILOT_wrapper)
```

The correct paths for the placeholder values can be found by running the following commands in your activated virtual environment:

- **To find `<path_to_python_include>`:**
  ```bash
  python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])"
  ```

- **To find `<path_to_numpy_include>`:**
  ```bash
  python3 -c "import numpy; print(numpy.get_include())"
  ```

Replace the placeholders in `CMakeLists.txt` with the respective outputs of these commands. After saving the file, delete the contents of the `build` directory (`rm -rf *`), and run the `cmake ..` and `make` commands again.

## 6. Install Visualization Tools (Optional)

To generate visualizations of the decision trees, Graphviz and pydot are required.

```bash
sudo apt-get install graphviz
pip install pydot
```

After completing these steps, the environment is fully configured to run all experiments and analyses presented in this paper.
