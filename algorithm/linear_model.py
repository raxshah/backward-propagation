from activation.gates import MultiplyGate, AddGate, Unit
import math, random

class Circuit:
    def __init__(self):
        self.mulg1 = MultiplyGate()
        self.mulg2 = MultiplyGate()
        self.addg1 = AddGate()
        self.addg2 = AddGate()

    def forward_pass(self,x,y,a,b,c):
        self.ax = self.mulg1.forward(a,x)
        self.by = self.mulg2.forward(b,y)
        self.ax_by = self.addg1.forward(self.ax,self.by)
        self.ax_by_c = self.addg2.forward(self.ax_by,c)
        return self.ax_by_c

    def backward_pass(self,gradient):
        self.ax_by_c.grad = gradient
        self.addg2.backward()
        self.addg1.backward()
        self.mulg2.backward()
        self.mulg1.backward()

class SVM:
    def __init__(self):
        self.a = Unit(1,0)
        self.b = Unit(-2,0)
        self.c = Unit(-1,0)
        self.circuit = Circuit()
    
    def forward_pass(self,x,y):
        self.op = self.circuit.forward_pass(x,y,self.a,self.b,self.c)
        return self.op

    def backward_pass(self,label):
        #reset backprop value
        self.a.grad = 0
        self.b.grad = 0
        self.c.grad = 0

        gradient = 0
        if label == 1 and self.op.val < 1:
            gradient = 1
        elif label == -1 and self.op.val > -1:
            gradient = -1

        self.circuit.backward_pass(gradient)

    def parameter_update(self):
        step_size = 0.01
        self.a.val = self.a.val + (step_size * self.a.grad - step_size * self.a.val) # after adding regularization
        self.b.val = self.b.val + (step_size * self.b.grad - step_size * self.b.val) # after adding regularization
        self.c.val = self.c.val + (step_size * self.c.grad)  # no regularization for bias

    def learn(self,x,y,label):
        self.forward_pass(x,y)
        self.backward_pass(label)
        self.parameter_update()

    def training_accuracy(self,data,labels):
        total_correct = 0
        for i in range(len(data)):
            x = Unit(data[i][0],0)
            y = Unit(data[i][1],0)
            true_label = labels[i]
            predicted_label = 1 if self.forward_pass(x,y).val > 0 else -1
            if true_label == predicted_label:
                total_correct += 1
        return total_correct/len(data)

    def train(self,data,labels):
        for i in range(400):
            #pick random data point
            index  = random.randint(0,len(data)-1)
            x = Unit(data[index][0],0)
            y = Unit(data[index][1],0)
            label = labels[index]
            self.learn(x,y,label)

            # check train loss after every 25 iterations
            if (i % 25) == 0:
                print('Train accuracy at iteration {} :'.format(i), self.training_accuracy(data,labels))

    def gradient_verify(self,a,b,c,x,y,label):
        def output_function(a,b,c,x,y):
            return a*x+b*y+c
        #anlytical gradient compute
        self.a = a
        self.b = b
        self.c = c
        self.learn(x,y,label)
        
        #numeric gradient compute
        h = 0.0001
        a_grad = (output_function(a.val+h,b.val,c.val,x.val,y.val) - output_function(a.val,b.val,c.val,x.val,y.val))/h
        b_grad = (output_function(a.val,b.val+h,c.val,x.val,y.val) - output_function(a.val,b.val,c.val,x.val,y.val))/h
        c_grad = (output_function(a.val,b.val,c.val+h,x.val,y.val) - output_function(a.val,b.val,c.val,x.val,y.val))/h

        #compare both gradient
        for i,j in zip([self.a,self.b,self.c],[a_grad,b_grad,c_grad]):
            print('Analytical gradient:',i.grad,'Numerical gradient:',j, sep= ' ')
