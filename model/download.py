import kagglehub
import os

sk_path = kagglehub.model_download("sudarshan1927/brain-tumor-ml-models/scikitLearn/default")
deep_path = kagglehub.model_download("sudarshan1927/brain-tumor-ml-models/pyTorch/deep-model")

print("Path to model files:", sk_path)
print("Path to model files:", deep_path)

bt_model = os.path.join(deep_path, 'BTModel.pth')
dense_model = os.path.join(deep_path, 'DenseNetModel.pth')
efnet_model = os.path.join(deep_path, 'EFNetModel.pth')
resnet_model  =os.path.join(deep_path, 'ResNet18Model.pth')
label_encoder = os.path.join(sk_path, 'label_encoder.joblib')
rf_classifier = os.path.join(sk_path, 'rf_classifier.joblib')
svm_classifier = os.path.join(sk_path, 'svm_classifier.joblib')
xgb_classifier = os.path.join(sk_path, 'xgb_classifier.joblib')
