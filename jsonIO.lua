local cjson = require 'cjson'

local M = {}

function M.read(path)
	local file = io.open(path, 'r')
	local text = file:read()
	file:close()
	local info = cjson.decode(text)
	return info
end


function M.write(path, j)
	cjson.encode_sparse_array(true, 2, 10)
	local text = cjson.encode(j)
	local file = io.open(path, 'w')
	file:write(text)
	file:close()
end

function M.preprocessing(data)
	L = data:numel()
	--[[
	return torch.cdiv(
	data:narrow(1,2,L-1) - data:narrow(1,1,L-1),
	data:narrow(1,1,L-1))
	]]--
	return  --log return
	torch.log(torch.cdiv(data:narrow(1,2,L-1),data:narrow(1,1,L-1))) 
end

function M.extractDataToTensor(ticker_table,tickers)
	local tickers = tickers or {'close', 'low', 'high'}
	local new = {}
	for junk,v in pairs(tickers) do
		new[v] = {}
	end
	--print(new)
	for i = 1,#ticker_table do
		for junk,v in pairs(tickers) do
			table.insert(new[v], ticker_table[i][v])
		end
	end
	--print(new.close)
	--[[
	for junk,v in pairs(tickers) do
		new[v] = M.preprocessing(torch.FloatTensor(new[v]))
	end
	]]--
	for i,v in ipairs(tickers) do
		if i>1 then
			out = torch.cat(out, M.preprocessing(torch.FloatTensor(new[v])),2)
		else
			--[[if type(new[v][1]) ~= 'number' then
				return nil
			end
			]]--
			out = M.preprocessing(torch.FloatTensor(new[v]))
		end
	end
	return out
end

function M.json2t7(json_file, t7_file, tickers_g)
	local tickers = tickers_g or {'close','low', 'high'}
	if not (type(tickers) == 'table') then
		local t = {}
		if tickers:find(',') then
			t = tickers:split(',')
		else
			table.insert(t, tickers)
		end
		tickers = t
	end
	print(tickers)
	local t = M.read(json_file)
	local new_t = {}
	local count,total,count2 = 0, 0, 0
	for k,v in pairs(t) do 
		total = total + 1
		ok,_ = pcall( function() return  t[k].prices[1] end)
		if ok then
			--print(k)
			ok, new_t[k] = pcall( function() return M.extractDataToTensor(t[k]['prices'], tickers) end)
			if ok then
				count = count + 1
			else
				print(k)
				new_t[k] = nil
			end
		else
			new_t[k] = nil
			print(k)
		end
	end
	print(('%d out of %d converted'):format(count,total))
	local L = new_t.AMZN:size(1)
	print('line segment size '..L)
	local count2 = 0
	for k,v in pairs(new_t) do
		if (type(new_t[k]) == 'torch.FloatTensor') and (new_t[k]:size(1) == L) then
			count2 = count2+1
		end
	end
	print(count2)
	torch.save(t7_file, new_t)
end

return M
