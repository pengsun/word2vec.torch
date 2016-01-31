--- google's word to vector 300 utility
-- w2vvocab: string -> index
-- v2wvocab: index -> string
-- M: matrix: |V|, 300
--

local opt = {
	binfilename = '/home/ps/data/googlenews/GoogleNews-vectors-negative300.bin',
	t7filename = '/home/ps/data/googlenews/word2vec.t7'
}

local VEC_SIZE = 300 -- dim 300 vector for each word

local function bintot7(opt)
	print('bintot7: read bin file from ' .. opt.binfilename)
	local file = torch.DiskFile(opt.binfilename,'r')
	local max_w = 50

	local function readStringv2(file)
		local str = {}
		for i = 1,max_w do
			local char = file:readChar()

			if char == 32 or char == 10 or char == 0 then
				break
			else
				str[#str+1] = char
			end
		end
		str = torch.CharStorage(str)
		return str:string()
	end
	print('bintot7: read header')
	file:ascii()
	local nwords = file:readInt()
	local size = file:readInt()

	local w2vvocab = {}
	local v2wvocab = {}
	local M = torch.FloatTensor(nwords,size)

	print('bintot7: read contents')
	file:binary()
	for i = 1, nwords do
		local str = readStringv2(file)
		local vecrep = file:readFloat(VEC_SIZE)
		vecrep = torch.FloatTensor(vecrep)
		local norm = torch.norm(vecrep,2)
		if norm ~= 0 then vecrep:div(norm) end
		w2vvocab[str] = i
		v2wvocab[i] = str
		M[{{i},{}}] = vecrep
		xlua.progress(i, nwords)
	end

	print('bintot7: writing t7 File for future usage.')
	local word2vec = {}
	word2vec.M = M
	word2vec.w2vvocab = w2vvocab
	word2vec.v2wvocab = v2wvocab
	torch.save(opt.t7filename,word2vec)

	return word2vec
end

local w2vutils = {}
if not paths.filep(opt.t7filename) then
	print('generating word2vec t7 data at ' .. opt.t7filename .. ' ...')
	w2vutils = bintot7(opt)
else
	print('reading word2vec t7 data at ' .. opt.t7filename .. ' ...')
	w2vutils = torch.load(opt.t7filename)
end

w2vutils.distance = function (self,vec,k)
	local k = k or 1	
	--self.zeros = self.zeros or torch.zeros(self.M:size(1));
	local norm = vec:norm(2)
	vec:div(norm)
	local distances, oldindex = torch.mv(self.M ,vec), nil
	distances , oldindex = torch.sort(distances,1,true)
	local returnwords = {}
	local returndistances = {}
	for i = 1,k do
		table.insert(returnwords, w2vutils.v2wvocab[oldindex[i]])
		table.insert(returndistances, distances[i])
	end
	return {returndistances, returnwords}
end

w2vutils.word2vec = function (self,word,throwerror)
   local throwerror = throwerror or false
   local ind = self.w2vvocab[word]
   if throwerror then
		assert(ind ~= nil, 'Word does not exist in the dictionary!')
   end
   return self.M[ind]
end

w2vutils.vec_size = function (self)
	return VEC_SIZE
end

return w2vutils
