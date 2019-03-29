from activation.gates import Unit
from algorithm import linear_model
import math, random


data = [(1.2,0.7),(-0.3, -0.5),(3.0, 0.1),(-0.1, -1.0),(-1.0, 1.1),(2.1, -3)]
labels = [1,-1,1,-1,-1,1]
model = linear_model.SVM()
model.train(data,labels)
print('Model Parameters:',end=' ')
print(model.a.val,model.b.val,model.c.val)


# # verify numeric gradient and anlytical gradient
# index  = random.randint(0,len(data)-1)
# x = Unit(1.2)
# y = Unit(0.7)
# a = Unit(1,0)
# b = Unit(-2,0)
# c = Unit(-1,0)
# label = 1
# model = linear_model.SVM()
# model.gradient_verify(a,b,c,x,y,label)



