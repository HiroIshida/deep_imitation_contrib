from deep_imitation import ConvAutoEncoderTrainer
from deep_imitation.data import get_epoch_summary
from deep_imitation.data import get_project_path

import os
import os.path as osp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

from sklearn.decomposition import PCA

class FeatureSpaceAnalyzer(object):
    def __init__(self, feature_seqs, project_name):
        self._feature_seqs = feature_seqs
        self._project_name = project_name

    @classmethod
    def cache_name(cls, project_name):
        project_path = get_project_path(project_name)
        cache_name = osp.join(project_path, 'feature_seq_cache.pickle')
        return cache_name

    @classmethod
    def from_pickled_summary(cls, project_name, **kwargs):
        cache_name = cls.cache_name(project_name)
        if os.path.exists(cache_name):
            return cls.from_direct_cache(project_name)

        data = get_epoch_summary(project_name)
        cae_trainer = ConvAutoEncoderTrainer.from_pickle(project_name)
        feature_seqs = data.to_feature_seqs(cae_trainer._model)
        feature_seqs_numpy = [np.array([f.detach().numpy() for f in fs]) for fs in feature_seqs]

        # make tmp cache
        with open(cache_name, 'wb') as f:
            pickle.dump(feature_seqs_numpy, f)
        return cls(feature_seqs_numpy, project_name)

    @classmethod
    def from_direct_cache(cls, project_name):
        project_path = get_project_path(project_name)
        cache_name = cls.cache_name(project_name)
        with open(cache_name, 'rb') as f:
            feature_seqs_numpy = pickle.load(f)
        return cls(feature_seqs_numpy, project_name)

if __name__=='__main__':
    fsa = FeatureSpaceAnalyzer.from_pickled_summary('robot')
    tmp = np.concatenate(fsa._feature_seqs)
    pca = PCA(n_components=3)
    pca.fit(tmp)


    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(100):
        X = pca.transform(fsa._feature_seqs[i])
        ax.plot(X[:, 0], X[:, 1], X[:, 2], linewidth=0.3)
    plt.show()



