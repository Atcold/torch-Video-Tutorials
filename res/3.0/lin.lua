-- Huge MLP

require 'nn'

local n = 256 * 256 * 3
local K = 1000
local net = nn.Sequential()

-- First layer
net:add(nn.Linear(n, 2*n))
net:add(nn.ReLU())
-- Second layer
net:add(nn.Linear(2*n, 2*n))
net:add(nn.ReLU())
-- Third layer
net:add(nn.Linear(2*n, 2*n))
net:add(nn.ReLU())
-- Output
net:add(nn.Linear(2*n, K))

return net
