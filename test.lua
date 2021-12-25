require 'nn'
require 'cunn'
require 'cudnn'

local dtype = 'torch.CudaTensor'
local json = require 'jsonIO'
local csv = require 'csvIO'
local cmd = torch.CmdLine()
-- opts for dataloader
cmd:option('-download', 0)
cmd:option('-csv_file', '')
cmd:option('-start_date', '', 'in YYYY-MM-DD format')
cmd:option('-end_date', '', 'in YYYY-MM-DD format')
cmd:option('-t7_file', 'data/latest.t7')
cmd:option('-json_file', 'data/latest.json')
--	cmd:option('-tickers', 'LYV,AMZN,CAT,CABO,TTWO,CMCSA,AMT,CSCO,IR,NFLX', 'stock tickers')
cmd:option('-channels', 'close,low,high,adjclose', 'channel of stock prices data to use')
cmd:option('-SPY_tickers', '^GSPC', 'ticker of SPY')
-- opts for checkpoints
cmd:option('-checkpoint', 'checkpoint/4c3t/nResBlock4nFeat128/epoch056.t7')


local function computeTarget(input_line_segment)
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
local function getTheDate()
	--6 weeks time interval
	local ti = (6*7)*24*60*60;
	local year = os.date('%Y')
	local month = os.date('%m')
	local date = os.date('%d')
	local end_date = year..'-'..month..'-'..date
	year = os.date('%Y',os.time()-ti)
	month = os.date('%m',os.time()-ti)
	date = os.date('%d', os.time()-ti)
	local start_date = year..'-'..month..'-'..date
	return start_date,end_date
end
function main()
	local opt = cmd:parse(arg)
	--load model
	print('loading the model ...')
	local checkpoint = torch.load(opt.checkpoint)
	local model = checkpoint.model
	checkpoint = nil
	model:evaluate()
	model:cuda()
	cudnn.convert(model, cudnn)
	print('getting the date ...')
	local start_date, end_date
	if opt.start_date ~= '' and opt.end_date ~= '' then
		start_date, end_date = opt.start_date, opt.end_date
	else
		start_date, end_date = getTheDate()
	end
	print(start_date)
	print(end_date)
	if opt.download > 0 then
		print('using python to download latest SP500 data in json format ...')
		--download latest sp500 data
		sys.execute( ('python yahoofinancials/downloader.py --ticker_csv data/latest_SP500.csv  --start_date %s  --end_date %s --json_file %s'):format(start_date, end_date, opt.json_file) )
		print('converting json to t7 format ...')
		json.json2t7(opt.json_file, opt.t7_file, opt.channels)	
	end
	print('clean up data and loading the following channels: ')
	print({opt.channels})

	local data = torch.load(opt.t7_file)
	local csv_table = {}
	csv_table[1] = {'Symbol','max-Classification','max-Confidence','min-Classification', 'min-Confidence'}
	print(csv_table[1])
	local num_channels = #(opt.channels:split(','))
	local line_size = data['AMZN']:size(1)
	local inputs = torch.CudaTensor(1,line_size,num_channels) 
	local output,inc
	if line_size%2 == 0 then
		inc = 4
	else
		inc = 5
	end
	--local weights = torch.FloatTensor({1/6,1/6,1/3,1/6,1/6}):cuda()
	local softmax = nn.SoftMax()
	--5 week to predict a week
	print('network predicting for all tickers ...')
	for k,v in pairs(data) do
		if v:size(1) == line_size then
			inputs:copy(data[k])
			local prob = 0
			for i = 0,4 do
				output = model:forward(inputs[{{},{i*2+1+inc,-1},{}}]):double()
				--print(output)
				if i == 2 then
					prob = prob + softmax:forward(output)*1/3
				else
					prob = prob + softmax:forward(output)*1/6
				end
			end
			local _, max_idx = torch.max(prob,2)
			max_idx = max_idx[1][1] --convert to number
			local _,min_idx = torch.min(prob,2)
			min_idx = min_idx[1][1]

			local tmp = {k, max_idx, prob[1][max_idx], min_idx, prob[1][min_idx]}
			table.insert(csv_table,tmp)

			collectgarbage()
			collectgarbage()
		else
			print(k)
		end
	end
	if opt.csv_file == '' then 
		opt.csv_file = 'predict/SP500_'..
		os.date('%Y')..os.date('%m')..os.date('%d')
		..'.csv'
	end
	print(('writing predictions to %s ...'):format(opt.csv_file))
	--print(csv_table)
	csv.write(opt.csv_file, csv_table)
end

main()
