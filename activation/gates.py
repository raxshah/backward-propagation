import math
class Unit:
    def __init__(self,val,grad=0):
        self.val = val
        self.grad = grad

class MultiplyGate:
    def __init__(self):
        pass

    def forward(self,u0,u1):
        self.u0 = u0
        self.u1 = u1
        activation = self.u0.val*self.u1.val
        self.utop = Unit(activation,0)
        return self.utop

    def backward(self):
        #local_gradient is uo for u1 and u1 for u0
        self.u0.grad += self.u1.val * self.utop.grad #local derivative * derivative of next layer (chain rule)
        self.u1.grad += self.u0.val * self.utop.grad

class AddGate:
    def __init__(self):
        pass

    def forward(self,u0,u1):
        self.u0 = u0
        self.u1 = u1
        activation = self.u0.val+self.u1.val
        self.utop = Unit(activation,0)
        return self.utop

    def backward(self):
        #local_gradient is 1 for both u0 and u1
        self.u0.grad += 1 * self.utop.grad #local derivative * derivative of next layer (chain rule)
        self.u1.grad += 1 * self.utop.grad

class SigmoidGate:
    def __init__(self):
        pass
    
    def sigmoid(self,value):
        return 1/(1+ math.exp(-value))
    
    #sigmoid gate only accepts one input
    def forward(self,u0):
        self.u0 = u0
        activation = self.sigmoid(self.u0.val)
        self.utop = Unit(activation,0)
        return self.utop

    def backward(self):
        #local_derivative = self.sigmoid(self.u0) * (1 -self.sigmoid(self.u0))
        s = self.sigmoid(self.u0.val) #local derivative * derivative of next layer (chain rule)
        self.u0.grad += (s*(1-s)) * self.utop.grad