import joblib
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.v2 as v2
import model.download as dd
from PIL import Image


classnames = ['pituitary', 'notumor', 'meningioma', 'glioma']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet18 model
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Remove the final classification layer
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Identity() # type: ignore
resnet18 = resnet18.to(device)



transform = v2.Compose([
    v2.Grayscale(num_output_channels=3),
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(degrees=10),
    v2.RandomAdjustSharpness(sharpness_factor=5, p=0.5),
    v2.RandomAutocontrast(p=0.5),
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ResNet18Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Model, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.base_model(x)

class DenseNetModel(nn.Module):
    
    def __init__(self, num_classes):
        super(DenseNetModel, self).__init__()
        # Load pre-trained DenseNet121 model
        self.base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # Replace the classifier with a new one for our number of classes
        num_ftrs = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        return x

class EFNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EFNetModel, self).__init__()
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )

    
    def forward(self, x):
        return self.base_model(x)


class BTModel(nn.Module):
    def __init__(self, num_classes):
        super(BTModel, self).__init__()

        self.features = nn.Sequential(
           nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
        
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



label_encoder = joblib.load(dd.label_encoder)
# rf_classifier = joblib.load(dd.rf_classifier)
svm_classifier = joblib.load(dd.svm_classifier)
xgb_classifier = joblib.load(dd.xgb_classifier)

resnet_model = ResNet18Model(len(classnames)).to(device)
resnet_model.load_state_dict(torch.load(dd.resnet_model,
                                            map_location=device,
                                            weights_only=False))

densenet_model = DenseNetModel(len(classnames)).to(device)
densenet_model.load_state_dict(torch.load(dd.dense_model,
                                         map_location=device,
                                         weights_only=False))

efnet_model = EFNetModel(len(classnames)).to(device)
efnet_model.load_state_dict(torch.load(dd.efnet_model,
                                           map_location=device,
                                           weights_only=False))

bt_model = BTModel(len(classnames)).to(device)
bt_model.load_state_dict(torch.load(dd.bt_model,
                                        map_location=device,
                                        weights_only=False))

def predict(image: Image.Image):

    image_tensor: torch.Tensor = transform(image) # type: ignore
    densenet_model.eval()
    efnet_model.eval()
    bt_model.eval()
    resnet18.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        resnet_class = torch.argmax(resnet_model(image_tensor))
        densenet_class = torch.argmax(densenet_model(image_tensor))
        efnet_class = torch.argmax(efnet_model(image_tensor))
        bt_class = torch.argmax(bt_model(image_tensor))
        
        image_numpy = resnet18(image_tensor)
        image_numpy = image_numpy.cpu().numpy()
    
    # rf_class = rf_classifier.predict(image_numpy)
    svm_class = svm_classifier.predict(image_numpy)
    xgb_class = xgb_classifier.predict(image_numpy)

    results = {
        'resnet': resnet_class.item(),
        'densenet': densenet_class.item(),
        'efnet': efnet_class.item(),
        'bt': bt_class.item(),
        # 'rf': rf_class[0],
        'svm': svm_class[0],
        'xgb': xgb_class[0]
    }

    predicted_classes = np.array(list(results.values()))
    unique_values, counts = np.unique(predicted_classes, return_counts=True)

    # # Find the index of the maximum count
    max_count_index = np.argmax(counts)

    # # Get the number that appears the most
    most_frequent_number = unique_values[max_count_index]
    
    predicted_class = label_encoder.inverse_transform([most_frequent_number])[0]
    results['resnet'] = label_encoder.inverse_transform([results['resnet']])[0]
    results['densenet'] = label_encoder.inverse_transform([results['densenet']])[0]
    results['efnet'] = label_encoder.inverse_transform([results['efnet']])[0]
    results['bt'] = label_encoder.inverse_transform([results['bt']])[0]
    # results['rf'] = label_encoder.inverse_transform([results['rf']])[0]
    results['xgb'] = label_encoder.inverse_transform([results['xgb']])[0]
    results['svm'] = label_encoder.inverse_transform([results['svm']])[0]
    results['final'] = predicted_class
    
    return results

