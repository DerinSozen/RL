import pickle
import numpy as np

dataset = pickle.load(open('play_log.pkl','rb'))
print(np.array(dataset)[0].shape)