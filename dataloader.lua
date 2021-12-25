require 'xlua'
local json = require 'jsonIO.lua' 
local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader:__init(opt)
	local data 
	ok,	data = pcall(function() return torch.load(opt.t7_file) end)
	if not ok then
		print('can not read from t7_file, trying json_file...')
		local t7_file = paths.concat(paths.dirname(opt.json_file), 'tmp.t7')
		json.json2t7(opt.json_file, t7_file, opt.channels)
		data = torch.load(t7_file)
	end
	
	self.nDataChannel = opt.nDataChannel
	self.nInputChannel = opt.nInputChannel
	self.tickers = {} --opt.tickers --a table of stock tickers that excludes SPY
	for k,v in pairs(data) do
		table.insert(self.tickers, k)
	end
	self.num_tickers = #self.tickers --exclude SPY index
	print('num of tickers', self.num_tickers)
	self.SPY_ticker = opt.SPY_ticker or '^GSPC'
	--self.SPY_data = data[self.SPY_ticker]
	--self.line_size = self.SPY_data:size(1)
	self.line_size = opt.line_size or data['AMZN']:size(1)
	self.num_val = opt.num_val_tickers or 1
	self.line_segment_size = opt.line_segment_size or 24
	self.split_idxs = {
		train = 1,
		val = 1,
	}
	self.split_size = {
		train = self.num_tickers-self.num_val,
		val = self.num_val,
	}
	self.batch_size = opt.batch_size
	self.perm = {
		train = torch.randperm(self.num_tickers-self.num_val),
		val = torch.randperm(self.num_val), 
	}
--	print(self.split_size)
	self.data = {
		train = torch.FloatTensor(self.split_size.train, self.line_size, self.nDataChannel),
		val = torch.FloatTensor(self.split_size.val, self.line_size, self.nDataChannel),
	}
	for i,v in ipairs(self.tickers) do
		if i <= self.split_size.train then
			self.data.train[i]:copy(data[v]) 
		else
			self.data.val[i-self.split_size.train]:copy(data[v])
		end
	end
	data = nil
	collectgarbage()
	collectgarbage()
end

function DataLoader:reset(split)
	self.split_idxs[split] = 1
	self.perm[split] = torch.randperm(self.split_size[split])
end

function DataLoader:computeTarget(input_line_segment)
	--'close' in the first channel
	assert( input_line_segment:size(1) == 5)
	-- extract first channel
	local tdata = input_line_segment:narrow(2,1,1):squeeze()
	local target, aid = 0, 0
	for i = 5,1,-1 do
		if tdata[i] > 0 then
			 aid = aid + 1
		end
	end
	--[[
	if (aid >= 2) and (aid <= 3) then
		if tdata:sum() > 0 then
			target = 2
		else
			target = 3
		end
	elseif (aid >= 0) and (aid <= 1) then
		target = 4
	else
		target = 1
	end
	]]--

	return target
end
--[[
function DataLoader:computeTarget(input_line_segment)
	--'close' in the first channel
	assert( input_line_segment:size(1) == 5)
	-- extract first channel
	local tdata = input_line_segment:narrow(2,1,1):squeeze()
	local target, aid = 0, 0
	for i = 5,1,-1 do
		if tdata[i] > 0 then
			 aid = aid + 1
		end
	end
	if ( tdata:sum()>0 ) and ( aid>3 ) then
		target = 1
	elseif ( tdata:sum()< 0 ) and ( aid<2 ) then
		target = 2
	else
		target = 3
	end
	return target
end
]]--
--[[
function DataLoader:computeTarget(input_line_segment)
	--'close' in the first channel
	assert( input_line_segment:size(1) == 5)
	-- extract first channel
	local tdata = input_line_segment:narrow(2,1,1):squeeze()
	local target, aid = 0, 1
	for i = 5,1,-1 do
		if tdata[i] > 0 then
			target = target + aid
		end
		aid = aid*2
	end
	return target+1
end
]]--
function DataLoader:getBatch(split)
	if split == 'train' then
		if self.batch_size > self.split_size[split] - self.split_idxs[split] + 1 then
			self:reset(split)
		end
		local start_idx = self.split_idxs[split]

		local indices = self.perm[split]:narrow(1, start_idx, self.batch_size)
		local input = torch.FloatTensor(self.batch_size, self.line_segment_size, self.nDataChannel)
		local target = torch.FloatTensor(self.batch_size)
		for i = 1, self.batch_size do
			--[[
			local _, t = table.splice(self.data[split], indices[i], 1)
			-- truncate line segments of length 29
			t = unpack(t) --convert single element table to tensor
			]]--
			local t = self.data[split][indices[i]]
			local l = torch.random(1,self.line_size - self.line_segment_size -5 + 1)
			t = t:narrow(1, l, self.line_segment_size+5)
			input[i]:copy(t:narrow(1,1,self.line_segment_size))
			--spy = self.SPY_data:narrow(1, l, self.line_segment_size)
			--input[i]:copy(torch.cat(t:narrow(1, 1, self.line_segment_size),spy,2))
			target[i] = self:computeTarget(t:narrow(1,self.line_segment_size+1,5))
		end
		
		self.split_idxs[split] = start_idx + self.batch_size
		--[[
		return {
			input = input,
			target = target,
		}
		]]--
		return {
			input = input:narrow(3,1,self.nInputChannel),
			target = target,
		}

	elseif split == 'val' then
		local stride = 16
		local idx = 1
		local nBatch = math.floor((self.line_size - self.line_segment_size - 5)/stride) + 1
		local input = torch.FloatTensor(self.num_val, nBatch, self.line_segment_size, self.nDataChannel)
		local target = torch.FloatTensor(self.num_val, nBatch)
		-- extract line segments with some stride
		for idx = 1, self.num_val do 
			--local t = table.splice(self.data[split], idx, 1)
			local t = self.data[split][idx]
			local start_idx = 1
			local jdx = 1
			while start_idx <= self.line_size - self.line_segment_size + 1 -5 do
				local raw = t:narrow(1, start_idx, self.line_segment_size+5)
				--local spy = self.SPY_data:narrow(1, start_idx, self.line_segment_size)
				--input[idx][jdx]:copy(torch.cat(raw:narrow(1,1,self.line_segment_size), spy, 2))
				input[idx][jdx]:copy(raw:narrow(1,1,self.line_segment_size))
				target[idx][jdx] = self:computeTarget(raw:narrow(1,self.line_segment_size+1,5))
				jdx = jdx + 1
				start_idx = start_idx + stride
			end
		end
		--[[
		return {
			input = input,
			target = target,
		}
		]]--
		return {
			input = input:narrow(4,1,self.nInputChannel),
			target = target,
		}
	end	
end

return M.DataLoader
