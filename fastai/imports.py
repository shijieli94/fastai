import bz2
import collections
import concurrent.futures
import csv
import functools
import glob
import hashlib
import inspect
import io
import itertools
import json
import math
import mimetypes
import multiprocessing
import numbers
import operator
import os
import pickle
import random
import re
import shutil
import string
import sys
import tarfile
import tempfile
import threading
import time
import types
import typing
import urllib
import warnings
import zipfile
from collections import Counter, OrderedDict, defaultdict, namedtuple
from collections.abc import Generator, Iterable, Iterator, MutableSequence, Sequence
from concurrent.futures import as_completed
from contextlib import contextmanager, redirect_stdout
from copy import copy, deepcopy
from datetime import datetime
from enum import Enum, IntEnum
from functools import partial, reduce
from itertools import dropwhile, starmap, takewhile, zip_longest
from multiprocessing import Lock, Process, Queue, queues
from numbers import Number, Real
from operator import attrgetter, itemgetter, methodcaller
from pathlib import Path
from pdb import set_trace
from textwrap import TextWrapper
from types import SimpleNamespace
from typing import Any, Callable, Optional, TypeVar, Union
from urllib.request import urlopen

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# External modules
import requests
import scipy
import yaml
from fastcore.all import *
from fastprogress.fastprogress import master_bar, progress_bar
from numpy import array, ndarray
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from scipy import ndimage

try:
    from types import MethodDescriptorType, MethodWrapperType, WrapperDescriptorType
except ImportError:
    WrapperDescriptorType = type(object.__init__)
    MethodWrapperType = type(object().__str__)
    MethodDescriptorType = type(str.join)
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    FunctionType,
    LambdaType,
    MethodType,
)

pd.options.display.max_colwidth = 600
NoneType = type(None)
string_classes = (str, bytes)
mimetypes.init()

# PyTorch warnings
warnings.filterwarnings("ignore", message=".*nonzero.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*grid_sample.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Distutils.*", category=UserWarning)


def is_iter(o):
    "Test whether `o` can be used in a `for` loop"
    # Rank 0 tensors in PyTorch are not really iterable
    return isinstance(o, (Iterable, Generator)) and getattr(o, "ndim", 1)


def is_coll(o):
    "Test whether `o` is a collection (i.e. has a usable `len`)"
    # Rank 0 tensors in PyTorch do not have working `len`
    return hasattr(o, "__len__") and getattr(o, "ndim", 1)


def all_equal(a, b):
    "Compares whether `a` and `b` are the same length and have the same contents"
    if not is_iter(b):
        return False
    return all(equals(a_, b_) for a_, b_ in itertools.zip_longest(a, b))


def noop(x=None, *args, **kwargs):
    "Do nothing"
    return x


def noops(self, x=None, *args, **kwargs):
    "Do nothing (method)"
    return x


def one_is_instance(a, b, t):
    return isinstance(a, t) or isinstance(b, t)


def equals(a, b):
    "Compares `a` and `b` for equality; supports sublists, tensors and arrays too"
    if one_is_instance(a, b, type):
        return a == b
    if hasattr(a, "__array_eq__"):
        return a.__array_eq__(b)
    if hasattr(b, "__array_eq__"):
        return b.__array_eq__(a)
    cmp = (
        np.array_equal
        if one_is_instance(a, b, ndarray)
        else operator.eq
        if one_is_instance(a, b, (str, dict, set))
        else all_equal
        if is_iter(a) or is_iter(b)
        else operator.eq
    )
    return cmp(a, b)


def pv(text, verbose):
    if verbose:
        print(text)
