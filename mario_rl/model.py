import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def get_decision_network(input_size, nb_action, feature_encoder="conv", recurrent_encoder="lstm"):
    """ 
    Instantiates a decision network from a feature encoder and recurrent encoder
    """
    feature_encoders = {
        "none": lambda x: x,
        "conv": lambda: get_conv_encoder(input_size),
        "resnet": lambda: None,
        "vgg": lambda: None
    }

    encoder, recurrent_input_size = feature_encoders[feature_encoder]()

    recurrent_encoders = {
        "none": lambda: nn.Sequential(
            encoder, nn.Linear(recurrent_input_size[-1], nb_action)),
        "lstm": lambda: Network(recurrent_input_size, nb_action, 
                                feature_encoder=encoder),
        "bert": lambda: None
    }
    return recurrent_encoders[recurrent_encoder]()


# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out)) # thanks to this initialization, we have var(out) = std^2
    return out

# Initializing the weights of the neural network in an optimal way for the learning
def weights_init(m):
    classname = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size()) #?? list containing the shape of the weights in the object "m"
        fan_in = weight_shape[1] # dim1
        fan_out = weight_shape[0] # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)

def get_conv_encoder(input_size):
    conv_encoder = torch.nn.Sequential(
        Preprocessor(),
        torch.nn.Conv2d(1, 32, kernel_size=(3, 3)),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(kernel_size=(2, 2)),
        torch.nn.Conv2d(32, 64, kernel_size=(3, 3)),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2)),
        torch.nn.Dropout(0.1),
        Flatten()
    ).cuda()

    test_input = torch.zeros(input_size[:2])[None, None]
    return conv_encoder, conv_encoder(test_input.cuda()).shape

class LSTMDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels=3, growth_rate=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        starting_side_length = torch.sqrt(torch.tensor(in_channels, dtype=torch.float))
        assert starting_side_length.ceil() == starting_side_length
        starting_side_length = starting_side_length.int()
        self.starting_shape = (starting_side_length, starting_side_length)
        self.in_channels = in_channels        
        
        self.relu = nn.ReLU(inplace=True)

        levels = []
        for level_num in range(num_levels):
            if level_num == num_levels - 1:
                current_out_channels = out_channels
            else:
                current_out_channels *= growth_rate^(num_levels - level_num)
            current_in_channels = growth_rate^(num_levels - level_num)
            levels.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=1, 
                    out_channels=current_out_channels, 
                    kernel_size=4, 
                    stride=1, 
                    padding=0, 
                    bias=False
                    ),
                    nn.BatchNorm2d(growth_rate*8),
                    self.relu
                    ))
            
        self.decoder = nn.Sequential(levels)

        self.out = nn.Tanh()

    def forward(self, x):
        x = x.reshape(self.starting_shape)
        decoded_state = self.decoder(x)
        return self.out(decoded_state)
    

class LSTMConvAttention(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv(
            in_channels=self.starting_shape, 
            out_channels=out_channels, 
            kernel_size=4)
    
    def forward(self, image, features):
        attention = F.tanh(self.conv(features))
        return image * F.interpolate(attention, image)



# Creating the architecture of the Neural Network
class Network(nn.Module): #inherinting from nn.Module
    
    #Self - refers to the object that will be created from this class
    #     - self here to specify that we're referring to the object
    def __init__(self, input_size, nb_action, feature_encoder):
        super(Network, self).__init__() #inorder to use modules in torch.nn
        # Input and output neurons

        self.encoder = feature_encoder

        self.lstm = nn.LSTMCell(input_size[-1], 32) # making an LSTM (Long Short Term Memory) to learn the temporal properties of the input
        self.fcL = nn.Linear(32, nb_action) # full connection of the
        self.apply(weights_init) # initilizing the weights of the model with random weights
        self.fcL.weight.data = normalized_columns_initializer(self.fcL.weight.data, 0.01) # setting the standard deviation of the fcL tensor of weights to 0.01
        self.fcL.bias.data.fill_(0) # initializing the actor bias with zeros
        self.lstm.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
        self.lstm.bias_hh.data.fill_(0) # initializing the lstm bias with zeros
        self.train() # setting the module in "train" mode to activate the dropouts and batchnorms
        self.lstm_state = None
        self.last_mask = None

        #self.lstm_conv_decoder = LSTMDecoder(64, 16)
        #self.attention = LSTMConvAttention(16, 3)

        #self.cuda()

    # For function that will activate neurons and perform forward propagation
    def forward(self, state):
        #if self.last_mask is not None:
        #    state = state * self.last_mask
        encoded_state = self.encoder(state)
        if self.lstm_state is not None and self.lstm_state[0].shape[0] != state.shape[0]:
            self.lstm_state = None
        # the LSTM takes as input x and the old hidden & cell states and ouputs the new hidden & cell states
        self.last_state, self.cell_state = self.lstm(encoded_state, self.lstm_state)
        self.lstm_state = (self.last_state, self.cell_state) 
        
        #decoded_last_state = self.lstm_conv_decoder()
        #self.last_mask = self.attention(state, decoded_last_state)

        return self.fcL(F.relu(self.last_state)) # returning the output of the actor (Q(S,A)), and the new hidden & cell states

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
