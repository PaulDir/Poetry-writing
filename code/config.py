import tensorflow as tf
import numpy as np
import argparse
import os
import random
import time
import collections

filename = "poetryTang.txt"

batch_size = 64
ratio = 0.8   



epochNum = 100                    # train epoch
