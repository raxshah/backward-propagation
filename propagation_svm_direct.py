import math, random


data = [[1.2, 0.7],[-0.3, -0.5],[3.0, 0.1],[-0.1, -1.0],[-1.0, 1.1],[2.1, -3]]
labels = [1,-1,1,-1,-1,1]

#train model here only (without using Wire and Circuit concept)
a =1 ; b =-2; c=-1

def training_accuracy(data,labels,a,b,c):
    total_correct = 0
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][1]
        true_label = labels[i]
        score = a*x + b*y +c
        predicted_label = 1 if score > 0 else -1
        if true_label == predicted_label:
            total_correct += 1
    return total_correct/len(data)

for i in range(25000//len(data)):
    #pick random data point
    index  = random.randint(0,len(data)-1)
    x = data[index][0]
    y = data[index][1]
    label = labels[index]
    
    score = a*x + b*y +c
    if label == 1 and score < 1:
        gradient = 1
    elif label == -1 and score > -1:
        gradient = -1
    else:
        gradient = 0
    
    step_size = 0.01
    a = a + (step_size * x * gradient - step_size * a)
    b = b + (step_size * y * gradient - step_size * b)
    c = c + (step_size * 1 * gradient)

    # check train loss after every 25 iterations
    if (i % 25) == 0:
        print('Train accuracy at iteration {} :'.format(i), training_accuracy(data,labels,a,b,c))

print('Model Parameters:',end=' ')
print(a,b,c)




