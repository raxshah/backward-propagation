from activation.gates import MultiplyGate
from activation.gates import SigmoidGate
from activation.gates import AddGate
from activation.gates import Unit
import math

#make each Wire(input) unit that holds two parameters; value and gradient
a = Unit(1,0)
b = Unit(2,0)
c = Unit(-3,0)
x = Unit(-1,0)
y = Unit(3,0)

mulg1 = MultiplyGate()
mulg2 = MultiplyGate()
addg1 = AddGate()
addg2 = AddGate()
sg1 = SigmoidGate()

#forward pass
def forward_pass():
    ax = mulg1.forward(a,x)
    by = mulg2.forward(b,y)
    ax_by = addg1.forward(ax,by)
    axbyc = addg2.forward(ax_by,c)
    op = sg1.forward(axbyc)
    return op

op = forward_pass()
print('First forward pass output:',op.val)

def backward_pass(op):
    op.grad = 1
    sg1.backward()
    addg2.backward()
    addg1.backward()
    mulg2.backward()
    mulg1.backward()
backward_pass(op)

def update_inputs(step_size=0.01):
    a.val += step_size * a.grad 
    b.val += step_size * b.grad
    c.val += step_size * c.grad
    x.val += step_size * x.grad
    y.val += step_size * y.grad
update_inputs()

#do one more forward pass to see if we are able to improve our output
print('Second forward pass output:',forward_pass().val)

#verify gradient with numerical gradient
def numerical_gradient():
    def output_function(a,b,c,x,y):
        return 1/(1+math.exp(-(a*x+b*y+c)))
    a,b,c,x,y = 1,2,-3,-1,3
    h = 0.0001
    a_grad = (output_function(a+h,b,c,x,y) - output_function(a,b,c,x,y))/h
    b_grad = (output_function(a,b+h,c,x,y) - output_function(a,b,c,x,y))/h
    c_grad = (output_function(a,b,c+h,x,y) - output_function(a,b,c,x,y))/h
    x_grad = (output_function(a,b,c,x+h,y) - output_function(a,b,c,x,y))/h
    y_grad = (output_function(a,b,c,x,y+h) - output_function(a,b,c,x,y))/h

    return a_grad,b_grad,c_grad,x_grad,y_grad

a_grad,b_grad,c_grad,x_grad,y_grad = numerical_gradient()

for i,j in zip([a,b,c,x,y],[a_grad,b_grad,c_grad,x_grad,y_grad]):
    print('Analytical gradient:',i.grad,'Numerical gradient:',j, sep= ' ')
