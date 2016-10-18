require 'nngraph'
require 'pretty-nn'
torch.manualSeed(0)

-- nn vs. nngraph
net = nn.Sequential();
net:add(nn.Linear(20, 10));
net:add(nn.Tanh());
net:add(nn.Linear(10, 10));
net:add(nn.Tanh());
net:add(nn.Linear(10, 1));
net

h1 = net.modules[1]()
h1
{h1}

net.modules[1].bias:view(1, -1)
h1.data.module.bias:view(1, -1)

h2 = net.modules[5](net.modules[4](net.modules[3](net.modules[2](h1))))
gNet = nn.gModule({h1}, {h2})

graph.dot(gNet.fg, 'mlp', 'mlp')
$ open mlp.svg

x = torch.randn(20)
net:forward(x)
gNet:forward(x)

function nn.gModule:getNode(id)
   for _, n in ipairs(self.forwardnodes) do
      if n.id == id then return n.data.module end
   end
   return nil
end

gNet:getNode(4)
{gNet:getNode(4)}

-- Dash - notation
g1 = - nn.Linear(20, 10)

g2 = g1
   - nn.Tanh()
   - nn.Linear(10, 10)
   - nn.Tanh()
   - nn.Linear(10, 1)
mlp = nn.gModule({g1}, {g2})

graph.dot(mlp.fg, 'mlp2', 'mlp2')
$ open mlp2.svg

-- Fancy architecture
input = - nn.Identity()
L1 = input
   - nn.Linear(10, 20)
   - nn.Tanh()
L2 = {input, L1}
   - nn.JoinTable(1)
   - nn.Linear(30, 60)
   - nn.Tanh()
L3 = {L1, L2}
   - nn.JoinTable(1)
   - nn.Linear(80, 1)
   - nn.Tanh()
g = nn.gModule({input},{L3})

graph.dot(g.fg, 'fancy', 'fancy')
$ open fancy.svg

-- A RNN example in nngraph
n = 3
K = 1
d = 5
nHL = 2
T = 4
xx = - nn.Identity()
hh1 = - nn.Identity()
hh2 = - nn.Identity()
h1 = {xx, hh1} - nn.JoinTable(1) - nn.Linear(n+d, d) - nn.Tanh()
h2 = {h1, hh2} - nn.JoinTable(1) - nn.Linear(2*d, d) - nn.Tanh()
yy = h2 - nn.Linear(d, K) - nn.Tanh()
rnn = nn.gModule({xx, hh1, hh2}, {h1, h2, yy})
x = torch.randn(n)
h0 = torch.zeros(d)
rnn:forward({x, h0, h0})
graph.dot(rnn.fg, 'myRnn', 'myRnn')
$ open myRnn.svg

-- Using https://github.com/e-lab/torch7-demos/tree/master/RNN-train-sample
RNN = require 'RNN'
timeNet, net = RNN.getModel(n, d, nHL, K, T)
graph.dot(net.fg, 'net', 'net')
$ open net.svg
net:forward({x, h0, h0})
graph.dot(net.fg, 'netFw', 'netFw')
$ open netFw.svg
graph.dot(timeNet.fg, 'timeNet', 'timeNet')
$ open timeNet.svg
