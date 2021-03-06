import cv2
import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
from pylab import *
from PIL import Image

# Get the path of the training set
image_path = "./test/NM- 707094961075260793431102.jpg"

# Load the classifier, class names, scaler, number of clusters and vocabulary 
im_features, image_paths, idf, numWords, voc = joblib.load("bof128.pkl")
print(image_paths)

# Create feature extraction and keypoint detector objects
detector = cv2.xfeatures2d.SIFT_create()
# fea_det = cv2.FeatureDetector_create("SIFT")
# des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []

im = cv2.imread(image_path)
kpts, des = detector.detectAndCompute(im, None)
    # kpts = fea_det.detect(im)
    # kpts, des = des_ext.compute(im, kpts)

# rootsift
#rs = RootSIFT()
#des = rs.compute(kpts, des)

des_list.append((image_path, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]

# 
test_features = np.zeros((1, numWords), "float32")
words, distance = vq(descriptors, voc)
for w in words:
    test_features[0][w] += 1

# Perform Tf-Idf vectorization and L2 normalization
test_features = test_features*idf
test_features = preprocessing.normalize(test_features, norm='l2')

score = np.dot(test_features, im_features.T)
rank_ID = np.argsort(-score)

# Visualize the results
figure()
gray()
subplot(6, 4, 1)
imshow(im[:, :, ::-1])
axis('off')
for i, ID in enumerate(rank_ID[0][0:20]):
    img = Image.open(image_paths[ID])
    gray()
    subplot(6,4,i+5)
    imshow(img)
    axis('off')

show()  
