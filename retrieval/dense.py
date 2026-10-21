import numpy as np
def rank(q_vec, p_vecs):  # one question vs its 10 passages
    return np.argsort(-(p_vecs @ q_vec))