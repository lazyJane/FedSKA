import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from torchvision.models import MobileNetV2


class Mclr(nn.Module):
    def __init__(self, input_dim = 784, hidden_dim = 64, output_dim = 10):
        super(Mclr, self).__init__()
        self.feature = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc_graph = nn.Linear(hidden_dim * 2, output_dim)
    
    def features(self, x):
        x = torch.flatten(x, 1) #把特征展平 28 × 28 = 784
        return self.feature(x)
    
    def linears(self, feature, graph=False, graph_encoding=None):
        if graph:
            #x = torch.cat([x, graph_encoding], dim=-1)
            #x = (x + graph_encoding)/2
            feature = graph_encoding
            x = self.fc(feature)
            #x = self.fc_graph(x)
        else:
            x = self.fc(feature)
        return x

    def forward(self, x, graph=False, graph_encoding=None):
        x = self.features(x)
        #print(x.shape) #torch.Size([32, 64])
        #print(graph_encoding.shape) #torch.Size([32, 784])
        x = self.linears(x, graph, graph_encoding)
        #x = torch.flatten(x, 1)
        #x = self.feature(x)
        #x = self.fc(x)
        results = {}
        results['output'] = F.log_softmax(x, dim=1) #使第二个维度的和为1
        results['logit'] = x
        return results

class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)

class DNN(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.last = nn.Linear(mid_dim, output_dim)
        
    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.last(x)
        #x = F.log_softmax(x, dim=1)
        results = {}
        results['output'] = F.log_softmax(x, dim=1) #使第二个维度的和为1
        results['logit'] = x
        return results
        return x


class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
     #输入层与卷积核需要有相同的通道数,输入层的channel与卷积核对应的channel进行卷积运算,然后每个channel的卷积结果按位相加得到最终的特征图
    """
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) #输入通道数，输出通道数（卷积核个数），卷积核尺寸 通道数就是特征图的个数
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5) #输入通道数，输出通道数（卷积核个数），卷积核尺寸

        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        # torch.Size([32, 1, 28, 28]) (Batch_size, channel, width, height)
        x = self.pool(F.relu(self.conv1(x))) # [32, 32, 24, 24] -> [32, 32, 12, 12]
        x = self.pool(F.relu(self.conv2(x))) # [32, 64, 8, 8] -> [32, 64, 4, 4]
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x)) # [32, 2048]
        x = self.output(x) # [32, 62]
        results = {}
        results['output'] = F.log_softmax(x, dim=1) #使第二个维度的和为1
        results['logit'] = x
        return results

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        feature = x
        x = self.output(x)
        return x
        results = {}
        results['output'] = F.log_softmax(x, dim=1) #使第二个维度的和为1
        results['logit'] = x
        results['feature'] = feature

'''
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.last = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last(x)
       
        #results = {}
        #results['output'] = F.log_softmax(x, dim=1) #使第二个维度的和为1
        #results['logit'] = x
        #results['feature'] = feature
        return x
'''

class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

       
        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )
        
        '''
        self.rnn = nn.GRU(
            input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers, dropout=0
        )
        '''
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward_encoder(self, x):
        encoded = self.encoder(x)
        
        h_encode, _ = self.rnn(encoded)
        return h_encode
    
    def forward_decoder(self, h_encode):
        output = self.decoder(h_encode)
        output = output.permute(0, 2, 1)
        return output


    def forward(self, input_):
        encoded = self.encoder(input_)
        #print(encoded.shape)#torch.Size([32, 80, 8])
        output, _ = self.rnn(encoded)#([32, 80, 256])
        #print(output.shape)
        #input()
        #print(_)
        output = self.decoder(output)
        #print(output.shape)#torch.Size([32, 80, 100])
        #input()
        #permute之前 
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T) 维度换位
        return output


def get_vgg11(n_classes):
    """
    creates VGG11 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

    return model


def get_squeezenet(n_classes):
    """
    creates SqueezeNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes

    return model


    
def get_mobilenet(n_classes, graph=False, graph_encoding=None):
    """
    creates MobileNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """

    '''
    x = self.features(x)
    # Cannot use "squeeze" as batch-size can be 1
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
    '''
    model = models.mobilenet_v2(pretrained=True)

    '''
    model.classifier
    Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=1280, out_features=1000, bias=True)
    )
    '''
        
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    '''
    results = {}
    results['output'] = F.log_softmax(x, dim=1) #使第二个维度的和为1
    results['logit'] = x
    return results
    '''
    return model

class MobilenetWithGraph(MobileNetV2):
    def __init__(self, n_classes, width_mult= 1.0, inverted_residual_setting= None, round_nearest= 8, block= None, norm_layer= None, dropout=0.5):
        super(MobilenetWithGraph, self).__init__(n_classes, width_mult, inverted_residual_setting, round_nearest, block, norm_layer, dropout) 
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, n_classes),
        )
        self.classifier_graph = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel*2, n_classes),
        )

    def features_flatten(self, feature):
        feature = nn.functional.adaptive_avg_pool2d(feature, (1, 1))
        feature = torch.flatten(feature, 1)

        return feature

    def linears(self, feature, graph=False, graph_encoding=None):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        
        if graph:
            #feature = torch.cat([feature, graph_encoding], dim=-1)
            feature = graph_encoding
            x = self.classifier(feature)
        else:
            # Cannot use "squeeze" as batch-size can be 1
            feature = self.features_flatten(feature)
            x = self.classifier(feature)
        
        return x
    
    def forward(self, x, graph=False, graph_encoding=None):
        x = self.features(x)
        x = self.linears(x, graph, graph_encoding)

        return x
    '''
    def forward(self, x, graph=False, graph_encoding=None):
        x = self.features(x)
        #print(x.shape) #torch.Size([32, 64])
        #print(graph_encoding.shape) #torch.Size([32, 784])
        x = self.linears(x, graph, graph_encoding)
        results = {}
        results['output'] = F.log_softmax(x, dim=1) #使第二个维度的和为1
        results['logit'] = x
        return results
    '''


def get_resnet18(n_classes):
    """
    creates Resnet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_resnet34(n_classes):
    """
    creates Resnet34 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model
