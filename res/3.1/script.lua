-- qlua -lenv
torch.manualSeed(0)
net = require '3conv-pool'
require 'image'

x = image.scale(image.lena(), 256, 256)
= #x
image.display{image = x, legend = 'img'}
= #net:forward(x)

-- qlua
= net:get(1)
-- th
net:get(1)
{net:get(1)}

-- Input / output planes
-- Stride
-- Kernel size
-- Check size of output
-- Check size of kernel / gradKernel
-- Check size of bias / gradBias

image.display{image = net:get(1).weight, legend = 'k(1)', zoom = 18, padding = 2}

function show(x, t)
   print(x)
   image.display{image = x.output, legend = t, scaleeach=true}
end

show(net:get(1), 'y(1)')
show(net:get(2), 'y(1)+')

show(net:get(3), 'y(2)')
show(net:get(4), 'y(2)+')
show(net:get(5), 'pool[y(2)+]')

show(net:get(6), 'y(3)')
show(net:get(7), 'y(3)+')
show(net:get(8), 'pool[y(3)+]')
