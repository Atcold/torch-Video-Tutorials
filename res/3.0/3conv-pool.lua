-- CNN with pooling layers

require 'nn'

local K = 1000
local net = nn.Sequential()

-- First layer
net:add(nn.SpatialConvolution(3, 6, 5, 5, 2, 2, 2, 2))
net:add(nn.ReLU())
-- Second layer
net:add(nn.SpatialConvolution(6, 6, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- Third layer
net:add(nn.SpatialConvolution(6, 6, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- MLP
net:add(nn.View(-1))

-- Prints the sizes of all layers output
-- x = torch.rand(3, 256, 256)
-- net:forward(x)
-- net:apply(function (m) print(m.output:size()) end)

-- Output
net:add(nn.Linear(6144, K))

return net
