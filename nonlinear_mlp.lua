-- Resources: 
-- * http://code.madbits.com/wiki/doku.php?id=tutorial_supervised_2_model
-- * https://github.com/torch/demos/blob/master/linear-regression/example-linear-regression.lua

require 'torch'
require 'optim'
require 'nn'

data = torch.Tensor{
	{0,0,1,0},
	{0,1,1,1},
	{1,0,1,1},
	{1,1,1,0}
}

model = nn.Sequential()
ninputs = 3; noutputs = 1; nhiddens = 4;
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs, nhiddens))
model:add(nn.Tanh())
model:add(nn.Linear(nhiddens, noutputs))

criterion = nn.MSECriterion()

x, dl_dx = model:getParameters()

feval = function(x_new)
	if x ~=x_new then
		x:copy(x_new)
	end

	_nidx_ = (_nidx_ or 0) + 1
	if _nidx_ > (#data)[1] then _nidx_ =1 end

	local sample = data[_nidx_]
	local target = sample[{ {4} }]
	local inputs = sample[{ {1,3} }]

	dl_dx:zero()

	-- See https://github.com/oxford-cs-ml-2015/practical4/blob/master/train.lua for expansion
	local loss_x = criterion:forward(model:forward(inputs), target)
	model:backward(inputs, criterion:backward(model.output, target))

	return loss_x, dl_dx
end

sgd_params = {
	learningRate = 1e-2,
	learningRateDecay = 1e-4,
	weightDecay = 0,
	momentum = 0
}

for i =1,1e4 do
	current_loss = 0
	for i = 1,(#data)[1] do
		_,fs = optim.sgd(feval, x, sgd_params)
		current_loss = current_loss + fs[1]
	end

	current_loss = current_loss / (#data)[1]
	print('current loss = ' .. current_loss)
end

for i = 1, (#data)[1] do
	local myPrediction = model:forward(data[i][{{1,3}}])
	print(string.format("%2d %6.2f", i, myPrediction[1]))
end