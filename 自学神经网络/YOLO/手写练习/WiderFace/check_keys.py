import scipy.io
mat = scipy.io.loadmat('wider_face_split/wider_face_train.mat')
print('所有顶层变量名：', list(mat.keys()))