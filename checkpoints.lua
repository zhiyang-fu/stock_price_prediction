--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local checkpoint = {}
local utils = require 'jsonIO'

local function deepCopy(tbl)
	-- creates a copy of a network with new modules and the same tensors
	local copy = {}
	for k, v in pairs(tbl) do
		if type(v) == 'table' then
			copy[k] = deepCopy(v)
		else
			copy[k] = v
		end
	end
	if torch.typename(tbl) then
		torch.setmetatable(copy, torch.typename(tbl))
	end
	return copy
end

function checkpoint.latest(opt)
	--[[if opt.resume == ''  or not opt.resume then
	if opt.checkpoint_dir then
	opt.resume = opt.checkpoint_dir
	else
	return nil
	end
	end
	]]--

	if not paths.dirp(opt.checkpoint_dir) then
		paths.mkdir(opt.checkpoint_dir)
	end
	local latestPath
	if opt.finetune then
		if paths.filep(opt.finetune) then
			latestPath = opt.finetune
		elseif paths.dirp(opt.finetune) then
			latestPath = paths.concat(opt.finetune, 'latest.t7')
		end
	else
		latestPath = paths.concat(opt.checkpoint_dir, 'latest.t7')
	end
	if opt.startover or not paths.filep(latestPath) then
		return nil, opt, nil, nil
	end

	print('=> Loading checkpoint ' .. latestPath)
	local latest_checkpoint = torch.load(latestPath)
	--local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))
	if opt.finetune then
		return nil, opt, latest_checkpoint.model, nil
	else -- resume training
		return latest_checkpoint, latest_checkpoint.opt, latest_checkpoint.model, latest_checkpoint.optim_state
	end
end

function checkpoint.save(epoch, model, optimState, best_epoch, opt, loss, rMSE)
	-- don't save the DataParallelTable for easier loading on other machines
	if torch.type(model) == 'nn.DataParallelTable' then
		model = model:get(1)
	end

	-- create a clean copy on the CPU without modifying the original network
	--model = deepCopy(model):float():clearState()
	model = model:clearState()


	--torch.save(paths.concat(opt.checkpoint_dir, checkpoint_file), checkpoint)
	torch.save(paths.concat(opt.checkpoint_dir, 'latest.t7'), {
		epoch = epoch,
		best_epoch = best_epoch,
		opt = opt,
		optim_state = optim_state,
		model = model,
		loss = loss,
	})
	utils.write(paths.concat(opt.checkpoint_dir, 'optsloss.json'), {
		opt = opt, 
		best_epoch = best_epoch,
		loss = loss,
	})
	if epoch == best_epoch then
		print('Found best epoch, saving model to checkpoint')
		local checkpoint_file = string.format('epoch%03d.t7',epoch)
		-- local optimFile = 'optimState_' .. epoch .. '.t7'
		local checkpoint  = {
			epoch = epoch,
			opt = opt,
			model = model,
			optim_state = optim_state,
		}
		torch.save(paths.concat(opt.checkpoint_dir, checkpoint_file), checkpoint)
	end
end

function checkpoint.plot(tbl, name, legend, xlabel,ylabel)
	-- Assume tbl is a table of numbers, or a table of tables
	local fig = gnuplot.pngfigure(paths.concat(name .. '.png'))
	-- local fig = paths.concat(name .. '.png')
	local legend = legend or name
	-- local logscale = logscale or false

	local function findMinMax(tb)
		local minKey, maxKey = math.huge, -math.huge    
		local minKeyValue, maxKeyValue
		for i = 1, #tb do
			if tb[i][1] < minKey then
				minKey = tb[i][1]
				minKeyValue = tb[i][2]
			end
			if tb[i][1] > maxKey then
				maxKey = tb[i][1]
				maxKeyValue = tb[i][2]
			end
		end
		return minKeyValue, maxKeyValue
	end

	local function typeTable(tb)
		for k, v in pairs(tb[1]) do
			if type(v) == 'table' then
				return 'table'
			else
				return 'number'
			end
		end
	end

	local function toTensor(tb)
		local xAxis = {}
		local yAxis = {}
		for i = 1, #tb do
			table.insert(xAxis, tb[i][1])
			table.insert(yAxis, tb[i][2])
		end
		return torch.Tensor(xAxis), torch.Tensor(yAxis)
	end

	local function __xAxisScale(xAxis)
		local maxIter = xAxis:max()
		if maxIter > 1e6 then
			return 1e6
		elseif maxIter > 1e3 then
			return 1e3
		else
			return 1
		end
	end

	local lines = {}
	local first, last
	local xAxisScale = 1
	if typeTable(tbl) ~= 'table' then -- single graph
		local xAxis, yAxis = toTensor(tbl)
		xAxisScale = __xAxisScale(xAxis)
		-- if logscale then
		--     yAxis:log():div(math.log(10))
		-- end
		table.insert(lines, {legend, xAxis:div(xAxisScale), yAxis, '-'})
		first, last = findMinMax(tbl)
	else -- multiple lines
		assert(type(legend) == 'table', 'legend must be a table, if you want to draw lines more than 1')
		local tmp, _ = toTensor(tbl[1])
		xAxisScale = __xAxisScale(tmp)
		for i = 1, #tbl do
			local xAxis, yAxis = toTensor(tbl[i])
			table.insert(lines, {legend[i], xAxis:div(xAxisScale), yAxis, '-'})
		end
		first, last = findMinMax(tbl[1])
	end
	--[[gnuplot.figure(100)
	gnuplot.raw('set ytics nomirror')
	gnuplot.raw('set y2tics')
	gnuplot.raw('set yrange [0.7:1.3]')
	]]--
	local ylabel = ylabel or 'Validation loss (a.u.)'  
	gnuplot.ylabel(ylabel)
	--[[if logscale then
	gnuplot.raw('set logscale y') 
	gnuplot.ylabel('log10(LR)')
	end
	]]--
	--gnuplot.raw('set format y "%.1t"')
	gnuplot.plot(lines)
	if first < last then
		gnuplot.movelegend('right', 'bottom')
	else
		gnuplot.movelegend('right', 'top')
	end
	gnuplot.grid(true)
	-- gnuplot.title(paths.basename(name))
	local xlabel = xlabel or 'Epochs'
	if xAxisScale > 1 then
		xlabel = xlabel .. ' (*1e' .. math.log(xAxisScale, 10) .. ')'
	end
	gnuplot.xlabel(xlabel)

	-- gnuplot.figprint(fig)
	gnuplot.plotflush(fig)
	gnuplot.closeall()
end

return checkpoint
