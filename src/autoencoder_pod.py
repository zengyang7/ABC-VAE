#!/usr/bin/env python
# Copyright 2018 YangZeng
""" Training the numeric data """

# standard library imports
import os, time, sys
import scipy.io as scio

# third party imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


mat_file = scio.loadmat(sys.argv[1])

parameters = mat_file['']

