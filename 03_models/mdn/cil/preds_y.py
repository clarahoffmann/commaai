import mdn
from mdn import sample_from_output
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
import numpy as np
import multiprocessing


preds = list(np.load('../../../../data/commaai/predictions/mdn/cil/preds.npy'))
#samples = list(np.load('../../../../data/commaai/predictions/mdn/cil/samples.npy'))

samples = []
no_samp = 1000
j = 0
for j in tqdm(range(0, 94570)):
    pred = preds[j]
    def sample_from_dist(i):
        return(np.apply_along_axis(sample_from_output, 1, pred, 1, 50, temp=1.0))
    with multiprocessing.Pool(10) as proc:
        y_samples = proc.map(sample_from_dist, np.array([i for i in range(0, no_samp)]))
    samples.append(np.array(y_samples).reshape(no_samp))
    if j % 1000 == 0:
        np.save('../../../../data/commaai/predictions/mdn/cil/samples.npy', np.array(samples))

np.save('../../../../data/commaai/predictions/mdn/cil/samples.npy', np.array(samples))