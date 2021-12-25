require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cutorch'
require 'gnuplot'

torch.setdefaulttensortype('torch.FloatTensor')
print('\n\n\n' .. os.date("%Y-%m-%d_%H-%M-%S") .. '\n')
local opts = require 'opts'
local opt = opts.parse(arg)
--opt.nChannel = 38
--opt.net_type = 'feataddBN'
--opt.auto_bnorm = true
--opt.test_every = 10000
--opt.num_epochs = 100
torch.manualSeed(opt.manualSeed)

local checkpoints = require 'checkpoints'
local models = require 'model/init'

local DataLoader = require 'dataloader'
local Trainer = require 'train'

--load previous checkpoint, if it exists
--local checkpoint, optim_state = nil, nil
--if not opt.startover then
local checkpoint, opt, model, optim_state = checkpoints.latest(opt)
--print(model:size())
--print(checkpoint)
--print(optim_state)
--end
print('loading model and criterion...')
local model, criterion = models.setup(opt, model, optim_state)
--local criterion = require 'loss/init'(opt)
--print(model)

print('Creating data loader...')
local loader = DataLoader(opt)
local trainer = Trainer(model, criterion, opt, optim_state)
--local json_file = paths.concat(opt.checkpoint_dir, 'opt_loss.json')

if opt.valOnly then
	print('Validate the model (at epoch ' .. opt.start_epoch  .. ')')
	local val_loss = trainer:test(opt.start_epoch, loader)
	print(string.format('Epoch %d: validate loss  %.4f', opt.start_epoch, val_loss))
else
	print('Train start')
	local start_epoch = checkpoint and checkpoint.epoch + 1 or 1
	local loss = checkpoint and checkpoint.loss or {train = {}, val = {}}
	local best_loss = loss.best_val_loss or math.huge
	local best_epoch = checkpoint and checkpoint.best_epoch or 1
	for epoch = start_epoch, opt.num_epochs do
		local train_loss = trainer:train(epoch, loader)
		local val_loss = trainer:test(epoch, loader)

		table.insert(loss.train,{epoch, train_loss})
		table.insert(loss.val,{epoch, val_loss})
		if val_loss < best_loss then
			best_loss = val_loss
			loss.best_val_loss = best_loss
			best_epoch = epoch
		end
		print( string.format('[Epoch %d] val loss = %.4g (Best ever: %.4g at epoch = %d)',
		epoch, val_loss, best_loss, best_epoch) )

		print('plotting at each epoch ...')
		-- train and val loss
		checkpoints.plot({loss.train, loss.val}, paths.concat(opt.checkpoint_dir,'lossEpoch'),{'train','val'})
		-- rMSE
		checkpoints.save(epoch, model, trainer.optim_state, best_epoch, opt, loss)
	end
	print(string.format(' * Finished best validate loss : %.4g', best_loss))
end


