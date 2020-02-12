class Parameter:
	def __init__(self, act_func, optim):
		self.activation_func = act_func
		self.optimizer = optim
		self.accuracy = 0
		self.confusion_matrix = []

	def __repr__(self):
		return f"Activation function: \
		{self.activation_func}, optimizer: {self.optimizer}, accuracy: {self.accuracy},\
		 conf_matrix: \n {self.confusion_matrix}."