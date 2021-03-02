import numpy as np

from simplegp.Nodes.BaseNode import Node


class AddNode(Node):

	def __init__(self):
		super(AddNode, self).__init__()
		self.arity = 2

	def __repr__(self):
		return '+'

	def _GetHumanExpressionSpecificNode(self, args):
		return '( ' + args[0] + ' + ' + args[1] + ' )'

	def _GetPytorchExpressionSpecificNode(self, args):
		return 'torch.add(' + args[0] + ',' + args[1] + ')'


class SubNode(Node):
	def __init__(self):
		super(SubNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '-'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' - ' + args[1] + ' )'

	def _GetPytorchExpressionSpecificNode(self, args):
		return 'torch.sub(' + args[0] + ',' + args[1] + ')'

class MulNode(Node):
	def __init__(self):
		super(MulNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '*'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' * ' + args[1] + ' )'


	def _GetPytorchExpressionSpecificNode(self, args):
		return 'torch.mul(' + args[0] + ',' + args[1] + ')'

class DivNode(Node):
	def __init__(self):
		super(DivNode,self).__init__()
		self.arity = 2

	def __repr__(self):
		return '/'

	def _GetHumanExpressionSpecificNode( self, args ):
		return '( ' + args[0] + ' / ' + args[1] + ' )'

	def _GetPytorchExpressionSpecificNode(self, args):
		return 'torch.div(' + args[0] + ',' + args[1] + ')'

class FeatureNode(Node):
	def __init__(self, id):
		super(FeatureNode,self).__init__()
		self.id = id

	def __repr__(self):
		return str(self.id)

	def _GetHumanExpressionSpecificNode( self, args ):
		return str(self.id)

	def _GetPytorchExpressionSpecificNode(self, args):
		return str(self.id)


class PowNode(Node):

	def __init__(self, power):
		super(PowNode, self).__init__()
		self.arity = 2

	def __repr__(self):
		return '^'

	def _GetHumanExpressionSpecificNode(self, args):
		return '( ' + args[0] + '**( ' + args[1] + ' ))'

	def _GetPytorchExpressionSpecificNode(self, args):
		return 'torch.pow(' + args[0] + ',' + args[1] + ')'


class ExpNode(Node):
	def __init__(self):
		super(ExpNode, self).__init__()
		self.arity = 1

	def __repr__(self):
		return 'exp'

	def _GetHumanExpressionSpecificNode(self, args):
		return 'exp( ' + args[0] + ' )'

	def _GetPytorchExpressionSpecificNode(self, args):
		return 'torch.exp(' + args[0] + ')'


class LogNode(Node):
	def __init__(self):
		super(LogNode, self).__init__()
		self.arity = 1

	def __repr__(self):
		return 'ln'

	def _GetHumanExpressionSpecificNode(self, args):
		return 'ln( ' + args[0] + ' )'

	def _GetPytorchExpressionSpecificNode(self, args):
		return 'torch.log(' + args[0] + ')'

class SumNode(Node):
	def __init__(self):
		super(SumNode, self).__init__()
		self.arity = 1

	def __repr__(self):
		return 'sum'

	def _GetHumanExpressionSpecificNode(self, args):
		return 'sum( ' + args[0] + ' )'

	def _GetPytorchExpressionSpecificNode(self, args):
		return 'torch.sum(' + args[0] + ')'

class AbsNode(Node):
	def __init__(self):
		super(AbsNode, self).__init__()
		self.arity = 1

	def __repr__(self):
		return 'abs'

	def _GetHumanExpressionSpecificNode(self, args):
		return 'abs( ' + args[0] + ' )'

	def _GetPytorchExpressionSpecificNode(self, args):
		return 'torch.abs(' + args[0] + ')'
