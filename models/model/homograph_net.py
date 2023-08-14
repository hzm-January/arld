import os.path
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from urllib.request import urlretrieve

BASELINE_WEIGHTS_PATH = 'https://github.com/baudm/HomographyNet/raw/master/models/homographynet_weights_tf_dim_ordering_tf_kernels.h5'
MOBILENET_WEIGHTS_PATH = 'https://github.com/baudm/HomographyNet/raw/master/models/mobile_homographynet_weights_tf_dim_ordering_tf_kernels.h5'


class HomographyNet(nn.Module):
    def init(self, use_weights=False):
        super(HomographyNet, self).init()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 8)
        )

        # if use_weights:
        #     weights_name = os.path.basename(BASELINE_WEIGHTS_PATH)
        #     weights_path, _ = urlretrieve(BASELINE_WEIGHTS_PATH, './models/' + weights_name)
        #     self.load_state_dict(torch.load(weights_path))
        def forward(self, input):
            x = self.conv_layers(input)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x