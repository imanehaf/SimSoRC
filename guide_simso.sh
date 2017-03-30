# How to use SimSo on CalculQuebec


# use python 2.7.2
module load python/2.7.2

# create virtual environment for python
# ref: https://wiki.calculquebec.ca/w/Python/en#Installing_modules
virtualenv exp1
source exp1/bin/activate

# install packages for python
# other packages may be required
pip install SimPy==2.3.1
pip install numpy
pip install pandas

# start working with SimSo
