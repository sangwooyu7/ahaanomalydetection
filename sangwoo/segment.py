import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scan import Scan
from receipt import Receipt
from read import read_receipts
from read import sample_receipts
from extract import extract_features
from collections import defaultdict

file_path = 'supermarket.csv'
receipts = read_receipts(file_path)
set_seed = 999
sample = sample_receipts(receipts, 0.1, set_seed)




