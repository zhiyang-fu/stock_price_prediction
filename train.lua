local optim = require 'optim'
local paths = require 'paths'
local sys = require 'sys'
require 'adam'
local M = {}
local Trainer = torch.class('mri.Trainer', M)
local dtype = 'torch.CudaTensor'

function Trainer:__init(model, criterion, opt, optim_state)
	--[[	model:type(dtype)
	if use_cudnn then
	cudnn.convert(model, cudnn)
	cudnn.benchmark = true --uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
	-- If this is set to false, uses some in-built heuristics that might not always be fastest.
	cudnn.fastest = true -- this is like the :fastest() mode for the Convolution modules,
	-- simply picks the fastest convolution algorithm, rather than tuning for workspace size
	cudnn.verbose = false -- this prints out some more verbose information useful for debugging
	end
	]]--
	self.model = model
	self.criterion = criterion

	self.opt = opt
	self.optim_state = optim_state or {learningRate = opt.learning_rate, learningRateDecay = 1/opt.lr_halflife, amsgrad = true, t = 0}
	self.optim = adam

	self.iter = nil --opt.test_every * (opt.start_epoch-1) + 1       --Total iterations

	--self.input = nil
	--self.target = nil
	self.params, self.grad_params = model:getParameters()
	self.current_loss = nil
	self.train_loss_ret = math.huge
	self.feval = function() return self.current_loss, self.grad_params end

	collectgarbage()
	collectgarbage()

	-- self.maxPerf, self.maxIdx = {}, {}
end

function Trainer:train(epoch, dataloader)
	--    local size = dataloader:size()
	local trainTimer = torch.Timer()
	local dataTimer = torch.Timer()
	local trainTime, dataTime = 0, 0
	local num_kept_batches, train_loss_acc, train_loss_print  = 0, 0, 0

	local pe = self.opt.print_every
	local te = self.opt.test_every
	self.iter = te*(epoch -1) + 1
	--	self.iter = self.optim_state.t  or te * (epoch-1)
	--	self.iter = self.iter + 1 -- avoid skip_batch at very first iteration

	cudnn.fastest = true
	cudnn.benchmark = true
	self.model:clearState()
	--self.model:training()
	--self.model:clearState()
	--self.params, self.gradParams = self.model:getParameters()
	collectgarbage()
	collectgarbage()

	--for n, batch in dataloader:run() do
	while true do
		batch = dataloader:getBatch('train')
		dataTime = dataTime + dataTimer:time().real

		--self:copyInputs(batch.input, batch.target, 'train')
		batch.input, batch.target = batch.input:type(dtype), batch.target:type(dtype)
		--dbg()
		------- CORE ------
		--self.model:zeroGradParameters()
		self.grad_params:zero()
		--dbg()
		self.model:forward(batch.input)
		self.current_loss = self.criterion:forward(self.model.output, batch.target)
		self.criterion:backward(self.model.output, batch.target)
		self.model:backward(batch.input, self.criterion.gradInput)
		------- CORE -----

		-- skip batch  --
		if self.opt.skip_batch < 0 or self.current_loss < self.train_loss_ret*self.opt.skip_batch then
			train_loss_print = train_loss_print + self.current_loss
			train_loss_acc = train_loss_acc + self.current_loss
			self.iter = self.iter + 1
			num_kept_batches = num_kept_batches + 1
			self:updateLR()
			self.optim(self.feval, self.params, self.optim_state)
		else
			print(('| Epoch %.3f / %d, Error is too large! Skip this batch. (Err: %.4g, epoch_loss: %.4g)')
			:format(self.iter/te, self.opt.num_epochs, self.current_loss, self.train_loss_ret)) 
		end

		trainTime = trainTime + trainTimer:time().real


		if self.iter % pe == 0 then
			print(string.format('| Epoch %.3f / %d, Time %.2f (Data: %.2f),  loss = %.4g',
			self.iter/te, self.opt.num_epochs, trainTime, dataTime, train_loss_print/pe), self.optim_state.learningRate)
			train_loss_print, trainTime, dataTime = 0, 0, 0
		end
		trainTimer:reset()
		dataTimer:reset()

		if self.iter % te == 0 then
			break
		end
	end
	self.train_loss_ret = train_loss_acc / num_kept_batches
	return self.train_loss_ret
end

function Trainer:updateLR()
	local iter, halfLife = self.iter, self.opt.lr_halflife
	local lr
	if self.opt.lr_decay == 'step' then --decay lr half periodically
		local nStep = math.floor( (iter-1)/halfLife )
		lr = self.opt.learning_rate / math.pow(2, nStep)
	elseif self.opt.lr_decay == 'exp' then -- decay lr exponentially. y = y0 * e^(-kt)
		local k = math.log(2) / halfLife
		lr = self.opt.learning_rate * math.exp(-l*iter)
	elseif self.opt.lr_decay == 'inv' then -- decay lr as y = y0 / ( 1+kt )
		local k = 1 / halfLife
		lr = self.opt.learning_rate / ( 1+k*iter )
	end

	self.optim_state.learningRate = lr
end


function Trainer:test(epoch, dataloader)
	--Code for multiscale learning
	local size = dataloader.__size
	local timer = torch.Timer()
	local iter = 0
	local val_loss_acc = 0

	cudnn.fastest = false
	cudnn.benchmark = false

	self.model:clearState()
	self.model:evaluate()
	collectgarbage()
	collectgarbage()

	--for n, batch in dataloader:run() do
	batch_all = dataloader:getBatch('val') 
	for n = 1, self.opt.num_val_tickers do
		local batch = {
			input = batch_all.input[n],
			target = batch_all.target[n],
		}
		batch.input, batch.target = batch.input:type(dtype), batch.target:type(dtype)
		--print(batch.input:type(), batch.target:size())
		local output
			output = self.model:forward(batch.input)
		val_loss_acc = val_loss_acc + self.criterion:forward(output, batch.target)
		
		iter = iter + 1

		self.model:clearState()
		output = nil

		collectgarbage()
		collectgarbage()
	end

	collectgarbage()
	collectgarbage()

	print(('[Epoch %d (iter/epoch: %d)] Test time: %.2f, Avg. loss: %.4f')
	:format(epoch, self.opt.test_every, timer:time().real, val_loss_acc/iter))

	self.model:training()
	return val_loss_acc/iter
end


return M.Trainer
