local cuda = false
require 'nn'

torch.manualSeed(1234)
local model = nn.Sequential()
local n = 2
local K = 1
local s = {n, 5, 5, K}
model:add(nn.Linear(s[1], s[2]))
model:add(nn.Tanh())
model:add(nn.Linear(s[2], s[3]))
model:add(nn.Tanh())
model:add(nn.Linear(s[3], s[4]))

local loss = nn.MSECriterion()

local m = 128
local X = torch.DoubleTensor(m, n) --CudaTensor
local Y = torch.DoubleTensor(m)    --CudaTensor

for i = 1, m do
   local x = torch.randn(2)
   local y = x[1] * x[2] > 0 and -1 or 1
   X[i]:copy(x) -- fine also for Cuda
   Y[i] = y     -- fine also for Cuda
end

if cuda then
   -- Running on GPU
   require 'cunn'
   model:cuda()
   loss:cuda()
   X = X:cuda()
   Y = Y:cuda()
end

local theta, gradTheta = model:getParameters()

local optimState = {learningRate = 0.15}

require 'optim'

for epoch = 1, 1e3 do
   function feval(theta)
      gradTheta:zero()
      local h_x = model:forward(X)
      local J = loss:forward(h_x, Y)
      print(J) -- just for debugging purpose
      local dJ_dh_x = loss:backward(h_x, Y)
      model:backward(X, dJ_dh_x) -- computes and updates gradTheta
      return J, gradTheta
   end
   optim.sgd(feval, theta, optimState)
end

print('Prev J: 0.1758')
net = model
