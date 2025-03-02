import numpy as np
import matplotlib.pyplot as plt
import time


#function that returns a predected value, I have considered the intercept as a constant
def pred(m,x):
    return m*x+intercept


#Generating the data with the formula y=mx+c+noise
#the input x consists of 100 random integers from 1 to 500
slope=6
intercept=197
x=np.random.randint(1, 501, 100)
y=slope*x + intercept + np.random.normal(0,1,(100,))


print("Linear Search")
plt.figure()
#Different slope values from -100 to 100 with increments of 0.01
m_values=np.linspace(-100,100,2001)
error=[]
start=time.time()
#Performing linear search by checking the loss with every m value we have
for m in m_values:
    #predicting the slope for every input data x that was generated
    y_pred = np.array([pred(m,i) for i in x])
    #calculating and storing the error
    error.append(np.mean((y-y_pred)**2))
end=time.time()
#printing the time taken to perform linear search
print(end-start)
#choosing the best m value
print(m_values[np.argmin(error)],error[np.argmin(error)])
#plotting the line versus the data
plt.xlim(1,100)
plt.ylim(min(y),max(y))
plt.plot(x,y)
plt.plot(pred(m_values[np.argmin(error)],x))
plt.xlabel("m")
plt.ylabel("L(m)")
plt.show()


print("Gradient Descent")
#choosing a learning rate
alpha = 0.000001
#choosing a random m value
m=np.random.randint(-100,100,1)[0]
print("Random M value:", m)
m_values=[]
error=[]
start=time.time()
#performing gradient descent for a 100 iterations
for i in range(100):
    y_pred=np.array([pred(m,i) for i in x])
    grad=(-2)*(np.mean(x*(y-(m*x)-intercept)))
    m=m-alpha*(grad)
    m_values.append(m)
    error.append(np.mean((y-y_pred)**2))
end=time.time()
#printing the time taken to perform gradient descent
print(end-start)
#printing the last m_value and error of gradient descent
print(m_values[len(m_values)-1],error[len(error)-1])
#plotting the line versus the data
plt.xlim(1,100)
plt.ylim(min(y),max(y))
plt.plot(x,y)
plt.plot(pred(m_values[len(m_values)-1],x))
plt.xlabel("m")
plt.ylabel("L(m)")
plt.show()
