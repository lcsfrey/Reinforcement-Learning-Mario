

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def get_decision_network(input_size, nb_action, 
                         feature_encoder="conv", 
                         recurrent_encoder="none",
                         recurrent_units=32, 
                         cuda=True):
    """ 
    Instantiates a decision network from a feature encoder and recurrent encoder
    """
    feature_encoders = {
        "none": lambda: nn.Identity(),
        "conv": lambda: get_conv_encoder(input_size),
        "resnet": lambda: None,
        "vgg": lambda: None
    }

    encoder, enc_output_size = feature_encoders[feature_encoder]()

    recurrent_encoders = {
        "none": lambda: (nn.Identity(), enc_output_size[-1]),
        "lstm": lambda: (LSTMEncoder(enc_output_size[-1], recurrent_units), recurrent_units), 
        "bert": lambda: None
    }

    recurrent_encoder, out_channels = recurrent_encoders[recurrent_encoder]()
    
    decision_network = Network(input_size, out_channels, nb_action, 
                               encoder, recurrent_encoder)
    if cuda:
        decision_network = decision_network.cuda()
    return decision_network


# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out)) # thanks to this initialization, we have var(out) = std^2
    return out


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16, activation="relu", **act_kwargs):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            get_activation(activation=activation, **act_kwargs),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x)

class ConvAttention(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=3, padding=1)
    
    def forward(self, features):
        attention = F.sigmoid(self.conv(features))
        return features * attention

def get_activation(activation, **act_kwargs):
    if activation == "relu":
        return nn.ReLU()
    if activation == "lrelu":
        return nn.LeakyReLU(**act_kwargs)
    if activation == "pelu":
        return nn.PReLU(**act_kwargs)
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "tanh":
        return nn.Tanh()


def get_conv_encoder(input_size, activation="pelu", **act_kwargs):
    conv_encoder = torch.nn.Sequential(
        Preprocessor(),
        torch.nn.Conv2d(1, 32, kernel_size=(5, 5)),
        SELayer(32, activation=activation),
        get_activation(activation, **act_kwargs),
        torch.nn.AvgPool2d(kernel_size=(2, 2)),
        torch.nn.Conv2d(32, 64, kernel_size=(3, 3)),
        get_activation(activation, **act_kwargs),
        SELayer(64, activation=activation),
        torch.nn.Conv2d(64, 64, kernel_size=(3, 3)),
        get_activation(activation, **act_kwargs),
        #torch.nn.Dropout(0.2),
        torch.nn.AvgPool2d(kernel_size=(2, 2)),
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3)),
        get_activation(activation, **act_kwargs),
        torch.nn.Dropout(0.25),
        ConvAttention(128),
        torch.nn.MaxPool2d(kernel_size=(2, 2)),
    ).cuda()

    test_input = torch.zeros(input_size[:2])[None, None]
    return conv_encoder, conv_encoder(test_input.cuda()).view(-1).shape


class LSTMEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTMCell(in_channels, out_channels)        
        self.lstm.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
        self.lstm.bias_hh.data.fill_(0) # initializing the lstm bias with zeros
        self.lstm_state = None
        self.out_channels = out_channels

    def forward(self, x):
        if self.lstm_state is not None and self.lstm_state[0].shape[0] != x.shape[0]:
            self.lstm_state = None
        # the LSTM takes as input x and the old hidden & cell states and ouputs the new hidden & cell states
        self.last_state, self.cell_state = self.lstm(x, self.lstm_state)
        self.lstm_state = (self.last_state, self.cell_state) 
        return F.relu(self.last_state)



# Creating the architecture of the Neural Network
class Network(nn.Module): #inherinting from nn.Module
    
    #Self - refers to the object that will be created from this class
    #     - self here to specify that we're referring to the object
    def __init__(self, input_size, out_channels, nb_action, feature_encoder, recurrent_decoder):
        super(Network, self).__init__() # inorder to use modules in torch.nn

        self.encoder = feature_encoder

        self.recurrent_decoder = recurrent_decoder

        self.action_decoder = nn.Linear(out_channels, nb_action)
        self.action_decoder.weight.data = normalized_columns_initializer(self.action_decoder.weight.data, 0.01) # setting the standard deviation of the action_decoder tensor of weights to 0.01
        self.action_decoder.bias.data.fill_(0) # initializing the actor bias with zeros

        self.train()

    # For function that will activate neurons and perform forward propagation
    def forward(self, state):
        # encode input images into latent state
        encoded_state = self.encoder(state)
        encoded_state = torch.flatten(encoded_state, start_dim=1)

        # apply recurrent stucture to encoded latent states
        recurrent_encoded_state = self.recurrent_decoder(encoded_state)

        # map recurrent state onto action space
        action_probs = self.action_decoder(recurrent_encoded_state)
        return action_probs

def rgb2gray(rgb):
    h = rgb.cuda() * torch.tensor([0.2989, 0.5870, 0.1140])[None, :, None, None].cuda()
    return h.sum(1, keepdim=True)

class Preprocessor(nn.Module):

    def forward(self, inputs, output_shape=(64, 64)):

        if inputs.dim() == 2:
            inputs = inputs.unsqueeze_(0).unsqueeze_(0)
        elif inputs.dim() == 3:
            inputs = inputs.unsqueeze_(0)

        if inputs.shape[-1] == 1 or inputs.shape[-1] == 3:
            dims = torch.cat([torch.arange(inputs.dim()-3), 
                              torch.as_tensor([-1, -3, -2])])
            inputs = inputs.permute(dims.tolist())           
        
        inputs = inputs.float()

        if len(inputs.shape) == 3:
            inputs = inputs[None]

        inputs = inputs.div(255)
        
        inputs = F.interpolate(inputs, size=output_shape)

        inputs = rgb2gray(inputs).float()

        return inputs
