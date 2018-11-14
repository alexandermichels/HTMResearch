# htm-timeseries

## Things to work on
* Category encoder and numeric encoder for SimpleSequence
* Moving Average Error (how does this time series act as a function of CPMC? does the MA get better faster for higher CPMC in all cases and how much faster?)
* Make the I/O better for the program

## Getting Started

We are using [NuPiC](https://github.com/alexandermichels/nupic) as a dependency which as specific requirements for some core packages like Numpy, so I highly recommend starting by making a [virtual environment with `virtualenv`](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).

Once you have that set up, we want to activate our virtual environment:

```bash
source <name of environment>/bin/activate
```

Then we can install all of our dependencies with:

```bash
pip install -r requirements.txt
```
