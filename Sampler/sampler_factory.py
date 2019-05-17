import abc
import numpy as np

class meta_sampler():
	"""Sampler for MAD adaptation"""
	__metaclass__ = abc.ABCMeta

	def __init__(self,blocks_to_fetch):
		"""
		Args:
			blocks_to_fetch: number of blocks to fetch for each call of the sample() function
		"""
		self._blocks_to_fetch = blocks_to_fetch
	
	@abc.abstractmethod
	def sample(self, distribution):
		"""
		Args:
			distribution: perform sampling from this distribution.
		"""
		pass

class fixed_sampler(meta_sampler):
	"""
	Return always the same fixed group, it does not perform sampling
	"""
	def __init__(self,blocks_to_fetch,fixed_id):
		"""
		Args:
			blocks_to_fetch: number of blocks to fetch for each call of the sample() function
			fixed_id: value to return when calling sample()
		"""
		super(fixed_sampler, self).__init__(blocks_to_fetch)
		self._fixed_id = fixed_id
	
	def sample(self,distribution):
		return [self._fixed_id]

class random_sampler(meta_sampler):
	"""
	Perform random sampling 
	"""
	def sample(self,distribution):
		return np.random.choice(range(distribution.shape[0]),size=self._blocks_to_fetch,replace=False)

class argmax_sampler(meta_sampler):
	"""
	Perform sampling according to an argmax of the sampling distribution
	""" 
	def sample(self,distribution):
		return np.argpartition(np.squeeze(distribution),-self._blocks_to_fetch)[-self._blocks_to_fetch:]

class sequential_sampler(meta_sampler):
	"""
	Sample block to train according to a round robin schema
	"""
	def __init__(self,blocks_to_fetch):
		super(sequential_sampler, self).__init__(blocks_to_fetch)
		self._sample_counter=0
	
	def sample(self,distribution):
		base_block = self._sample_counter%distribution.shape[0]
		result = [(base_block+i)%distribution.shape[0] for i in range(self._blocks_to_fetch)]
		self._sample_counter+=1
		return result

class probabilistic_sampler(meta_sampler):
	"""
	Sample according to the current probability distribution 
	"""
	def sample(self,distribution):
		return np.random.choice(range(distribution.shape[0]),size=self._blocks_to_fetch,replace=False,p=np.squeeze(distribution))

###############################################################################
SAMPLER_FACTORY = {
	'FIXED':fixed_sampler,
	'RANDOM':random_sampler,
	'ARGMAX':argmax_sampler,
	'SEQUENTIAL':sequential_sampler,
	'PROBABILITY':probabilistic_sampler
}

AVAILABLE_SAMPLER = SAMPLER_FACTORY.keys()

def get_sampler(name, blocks_to_fetch, fixed_id=0):
	assert(name in AVAILABLE_SAMPLER)
	if name=='FIXED':
		return SAMPLER_FACTORY[name](blocks_to_fetch,fixed_id)
	else:
		return SAMPLER_FACTORY[name](blocks_to_fetch)