# Installation


The code was tested on Ubuntu 20.04, with [Anaconda](https://www.anaconda.com/download) Python 3.10.12, CUDA 11.6, and [PyTorch]((http://pytorch.org/)) v1.13.
It should be compatible with PyTorch <=1.13 and python >=3.10 
After installing Anaconda:

0. [Optional but highly recommended] create a new conda environment. 

    ~~~
    conda create --name smart python=3.10
    ~~~
    And activate the environment.
    
    ~~~
    conda activate smart
    ~~~

1. Install PyTorch:

    ~~~
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    ~~~
    

2. Install necessary build libraries:

    ~~~
    conda install cython
    conda install ninja
    conda install cudatoolkit-dev -c conda-forge
    pip install python-dev-tools --user --upgrade
    numpy 1.23.5 !!!!!!!!!!!!!!!!! pip install numpy==1.23.5
    ~~~

3. Clone this repo:

    ~~~
    SMART_ROOT=/path/to/clone/SMART
    git clone https://github.com/sompt22/SMART.git 
    $SMART_ROOT
    ~~~

4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
    
5. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/)).

    ~~~
    cd $SMART_ROOT/src/lib/model/networks/
    cd DCNv2
    ./make.sh
    ~~~
