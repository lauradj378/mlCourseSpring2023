import torch
import torch.nn as nn


class Shallow(nn.Module):
    """
    Shallow feed forward network.
    """
    def __init__(self, in_dim, out_dim, width):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=width, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=width, out_features=out_dim, bias=False),
        )

    def forward(self, x):

        return self.layers(x)
    
    
class Dense(nn.Module):
        """
        Fully Connected feed forward network.

        `depth` is the total number of linear layers.
        """

        def __init__(self, in_dim, out_dim, width, depth):
            
            super().__init__()
            
            self.layers = []
            self.layers.append(nn.Linear(in_features=in_dim,out_features=width,bias=True))
            self.layers.append(nn.ReLU())
            for i in range(depth):
                self.layers.append(nn.Linear(in_features=width,out_features=width,bias=True))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(in_features=width,out_features=out_dim,bias=False))
            self.layers = nn.Sequential(*self.layers)
            #print(self.layers)
            #print(type(self.layers))
            
        def forward(self,x):
               
            return self.layers(x)    
        
        
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
# =============================================================================
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=(5,5), padding=2,padding_mode = 'reflect'),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size = (2,2),stride = 2))
# =============================================================================
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=(5,5), padding=2,padding_mode = 'reflect')
        self.relu1 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size = (2,2),stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size=(5,5))
        self.relu2 = nn.ReLU()
# =============================================================================
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size=(5,5)),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size = (2,2),stride = 2))
# =============================================================================
        self.layer3 = nn.Sequential(
                      nn.LazyLinear(120),
                      nn.ReLU(),
                      nn.LazyLinear(84),
                      nn.ReLU(),
                      nn.LazyLinear(10))
        
    def forward(self, x):
# =============================================================================
#         y = self.layer1(x)
# =============================================================================
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.avgpool(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.avgpool(y)        
        y = torch.flatten(y,start_dim=1) #flattens indices 1,2,3 (leaves 0 alone)
        y = self.layer3(y)
        return y
  
    
class LeNet5Modified(nn.Module):
    def __init__(self, n_classes, normalize:bool, dropout:float):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=(5,5), padding=2,padding_mode = 'reflect'),
            #nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = (2,2),stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size=(5,5)),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = (2,2),stride = 2))
        self.layer3 = nn.Sequential(
                      nn.Linear(400,120),
                      nn.Dropout(dropout),
                      nn.ReLU(),
                      nn.Linear(120,84),
                      nn.ReLU(),
                      nn.Linear(84,n_classes))
        
        if normalize:
            normalization_layers = nn.ModuleList()
            normalization_layers.append(nn.BatchNorm2d(num_features=6))
            normalization_layers.append(nn.BatchNorm2d(num_features=16))
            self.layer1 = nn.Sequential(
                self.layer1[0], normalization_layers[0], self.layer1[1], self.layer1[2]
            )
            self.layer2 = nn.Sequential(
                self.layer2[0], normalization_layers[1], self.layer2[1], self.layer2[2]
            )
        
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = torch.flatten(y,start_dim=1) #flattens indices 1,2,3 (leaves 0 alone)
        y = self.layer3(y)
        return y
    