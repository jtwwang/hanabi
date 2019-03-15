import pickle
import sys
sys.path.append("..")

import third_party

with open("sum_tree_ckpt.1850","rb") as f:
    w = pickle.load(f)

pickle.dump(w, open("sum_tree_ckpt.1850_py2.pkl","wb"), protocol=2)

