--issues: nn.BatchNormalization only works for 2D data
--TemporalConvolution does not support padding, 
--the input to TemporalConvolution is NxLxC, (BatchxLengthxChannel)
local nn = require 'nn'
require 'cunn'

local tconv = nn.TemporalConvolution
local avg = nn.Mean
local max = nn.TemporalMaxPooling
local tBN = nn.BatchNormalization
local relu = nn.ReLU
local seq = nn.Sequential
local concat = nn.ConcatTable
local id = nn.Identity
local cadd = nn.CAddTable
local mul = nn.Mul
local mulc = nn.MulConstant

function addSkip(model)
	local model = seq()
	:add(concat()
	:add(model)
	:add(id()))
	:add(cadd(true))
	return model
end

function conv(nInput, nOutput, k, s, p) 
	-- temporal convolution that support padding
	-- realize by using nn.Padding()
	local pLeft = nn.Padding(2, -1*p)
	local pRight = nn.Padding(2, p)
	if p>0 then
		return seq()
		:add(pLeft)
		:add(pRight)
		:add(tconv(nInput, nOutput, k, s))
	else
		return tconv(nInput, nOutput, k, s)
	end
end

local function resBlock(nFeat, addBN, scaleRes, ipMulc)
	local nFeat = nFeat or 64
	local addBN = addBN or false
	local scaleRes = (scaleRes and scaleRes ~=1) and scaleRes or false
	local ipMulc = ipMulc or true
	if addBN then
		return addSkip(
		seq()
		:add(conv(nFeat, nFeat, 3, 1, 1))
		:add(tBN(nFeat))
		:add(relu(true))
		:add(conv(nFeat, nFeat, 3, 1, 1))
		:add(tBN(nFeat))
		)
	else
		if scaleRes then
			return addSkip(
			seq()
			:add(conv(nFeat, nFeat, 3, 1, 1))
			:add(relu(true))
			:add(conv(nFeat, nFeat, 3, 1, 1))
			:add(mulc(scaleRes, ipMulc))
			)
		else
			return addSkip(
			seq()
			:add(conv(nFeat, nFeat, 3, 1, 1))
			:add(relu(true))
			:add(conv(nFeat, nFeat, 3, 1, 1))
			)
		end
	end
end

function createModel(opt)
	nFeat = opt.nFeat or 64
	nInputChannel = 3 --opt.nInputChannel 
	nOutputChannel = opt.nOutputChannel 
	nResBlock = opt.nResBlock
	addBN = opt.addBN or false
	scaleRes = opt.scaleRes or 0.1
	ipMulc = opt.ipMulc or true

	local body1 = seq()
	for i = 1, nResBlock do 
		body1:add(resBlock(nFeat, addBN, scaleRes, ipMulc))
	end

	local body2 = seq()
	for i = 1, nResBlock do 
		body2:add(resBlock(2*nFeat, addBN, scaleRes, ipMulc))
	end
	
	local body3 = seq()
	for i = 1, nResBlock do 
		body3:add(resBlock(4*nFeat, addBN, scaleRes, ipMulc))
	end

	model = seq()
	:add(conv(nInputChannel, nFeat, 3, 1, 1)) --input conv
	:add(body1) -- ResNet at 24 
	:add(conv(nFeat, nFeat*2, 4, 2, 1)) -- downsampling
	:add(body2) -- ResNet at 12
	:add(conv(nFeat*2, nFeat*4, 4, 2, 1)) --downsampling
	:add(body3) -- ResNet at 6
	:add(avg(2,3)) --avg pooling
	:add(nn.Linear(nFeat*4, nOutputChannel))
	return model
end

return createModel
