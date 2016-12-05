# this function calculate the MAP score of sample.
class MapScore():
	def __init__(self):
		self.N = 0
		self.score = 0
		# target is the true answer of the sample
		# and predict is the sequence you predict
	def update_score(self, target, predict):
		if target in predict:
			self.score+=float(1)/(predict.index(target)+1) 
		self.N+=1
	def output_score(self):
		return self.score/self.N
	def clean(self):
		self.N, self.score = 0, 0
