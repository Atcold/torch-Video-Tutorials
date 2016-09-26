local net = nn.Sequential()

-- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.SpatialConvolution(1, 6, 5, 5))

-- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialMaxPooling(2,2,2,2))

-- non-linearity
net:add(nn.Tanh())

-- additional layers
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.Tanh())

-- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.View(16*5*5))

-- fully connected layers (matrix multiplication between input and weights)
net:add(nn.Linear(16*5*5, 120))
net:add(nn.Tanh())
net:add(nn.Linear(120, 84))
net:add(nn.Tanh())

-- 10 is the number of outputs of the network (10 classes)
net:add(nn.Linear(84, 10))
return net
