local M = {}

function M.parse(arg)
	local cmd = torch.CmdLine()
-- opts for dataloader
	cmd:option('-t7_file', '')
	cmd:option('-json_file', '')
--	cmd:option('-tickers', 'LYV,AMZN,CAT,CABO,TTWO,CMCSA,AMT,CSCO,IR,NFLX', 'stock tickers')
	cmd:option('-channels', 'close,low,high,adjclose', 'channel of stock prices data to use')
	cmd:option('-SPY_tickers', '^GSPC', 'ticker of SPY')
	cmd:option('-num_val_tickers', 1, 'number of tickers for validate')
	cmd:option('-line_size', 0, 'number of temporal points (daily) for each ticker' )
	cmd:option('-line_segment_size', 24, 'number of temporal points (daily) used to estimate the data of next 5 days' )
	cmd:option('-nInputChannel', 3, 'close, low, high of the stock')
	cmd:option('-nDataChannel', 6, 'close, low, high and SPY compose the 3*2 channels')
	cmd:option('-batch_size', 4)
-- opts for checkpoints
	cmd:option('-checkpoint_dir', '')
	cmd:option('-finetune', -1)
	cmd:option('-start_over', 1)
	cmd:option('-resume', 1)

-- generic opts
	cmd:option('-manualSeed', 0, 'manually set rng seed')
	cmd:option('-gpu', 0)
	cmd:option('-use_cudnn', 1)
-- opts for model
	cmd:option('-net_type', 'resnet1dD')
	cmd:option('-nResBlock', 4, 'number of residual blocks at each resolution')
	cmd:option('-nFeat', 64, 'number of features for basis resblock')
	cmd:option('-scaleRes', 0.1, 'constant scaling scalar inside resblock')
	cmd:option('-ipMulc', 1, 'in place mul for scaleRes')
	cmd:option('-addBN', -1, 'currently batch normalization for 1D in batch mode')
	cmd:option('-nOutputChannel', 32, 'predict one week data, leads to 2^5 patterns of stock trend')
-- opts for training
	cmd:option('-print_every', 100, 'number of iterations to print')
	cmd:option('-test_every', 10000, 'number of iterations to test')
	cmd:option('-skip_batch', 32, 'skip training batch that has irregular high training loss')
	cmd:option('-num_epochs', 100, 'number of epochs to train')
	--cmd:option('-start_epoch', 1)
	cmd:option('-lr_halflife', 2e5, 'number of half life of learning rate')
	cmd:option('-lr_decay', 'step', 'type of learning rate decay: step, exp, inv')
	cmd:option('-learning_rate', 1e-4, 'initial learning rate')



	local opt = cmd:parse(arg or {})
--[[	local tickers = {}
	if opt.tickers:find(',') then
		tickers = opt.tickers:split(',')
	else
		table.insert(tickers, opt.tickers)
	end
	opt.tickers = tickers
	]]--
	opt.addBN = (opt.addBN > 0) 
	opt.ipMulc = (opt.ipMulc > 0)
	opt.finetune = (opt.finetune > 0)
	opt.resume = (opt.resume > 0)
	opt.start_over = (opt.start_over > 0)

	return opt
end

return M
