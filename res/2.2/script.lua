-- Recording script, not a runnable script

require 'nn';
lin = nn.Linear(5, 3)
lin
{lin}
lin.weight
lin.bias
Theta_1 = torch.cat(lin.bias, lin.weight, 2) -- New Tensor
Theta_1
lin:zeroGradParameters()

sig = nn.Sigmoid()
{sig}
require 'gnuplot';
z = torch.linspace(-10, 10, 21)
gnuplot.plot(z, sig:forward(z))
-- Forward pass
x = torch.randn(5)
a1 = x
h_Theta = sig:forward(lin:forward(x)):clone()
z2 = Theta_1 * torch.cat(torch.ones(1), x, 1)
a2 = z_1:clone():apply(function (z) return 1/(1 + math.exp(-z)) end)

-- Backward pass
loss = nn.MSECriterion()
? nn.MSECriterion
loss
loss.sizeAverage = false
y = torch.rand(3)
-- forward(input, target)
E = loss:forward(h_Theta, y)
E
(h_Theta - y):pow(2):sum()

dE_dh = loss:updateGradInput(h_Theta, y):clone()
dE_dh
2 * (h_Theta - y)

delta2 = sig:updateGradInput(z2, dE_dh)
dE_dh:clone():cmul(a2):cmul(1 - a2)

lin:accGradParameters(x, delta2)
{lin}
lin.gradWeight
lin.gradBias
delta2:view(-1, 1) * torch.cat(torch.ones(1), x, 1):view(1, -1)

lin_gradInput = lin:updateGradInput(x, delta2)
lin.weight:t() * delta2

net = nn.Sequential()
net:add(lin);
net:add(sig);
net

-- While true
pred = net:forward(x)
pred
h_Theta
err = loss:forward(pred, y)
err
E
gradCriterion = loss:backward(pred, y)
gradCriterion
dE_dh
net:zeroGradParameters()
net:get(1)
torch.cat(net:get(1).gradBias, net:get(1).gradWeight, 2)

oldWeight = net:get(1).weight:clone()
oldBias = net:get(1).bias:clone()
etha = 0.01
net:updateParameters(etha)
net:get(1).weight
oldWeight - 0.01 * net:get(1).gradWeight
net:get(1).bias
oldBias - 0.01 * net:get(1).gradBias

-- X: design matrix
-- Y: labels / targets matrix / vector

for i = 1, m do
   local pred = net:forward(X[i])
   local err = criterion:forward(pred, Y[i])
   local gradCriterion = criterion:backward(pred, Y[i])
   net:zeroGradParameters()
   net:backward(X[i], gradCriterion)
   net:updateParameters(learningRate)
end

for i = 1, m, batchSize do
   net:zeroGradParameters()
   for j = 0, batchSize - 2 do
      if i+j > m then break end
      local pred = net:forward(X[i+j])
      local err = criterion:forward(pred, Y[i+j])
      local gradCriterion = criterion:backward(pred, Y[i+j])
      net:backward(X[i+j], gradCriterion)
   end
   net:updateParameters(learningRate)
end

dataset = {}
function dataset:size() return m end
for i = 1, m do dataset[i] = {X[i], Y[i]} end
trainer = nn.StochasticGradient(net, loss)
trainer:train(dataset)

