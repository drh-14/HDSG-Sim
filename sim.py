import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

#Network of Agents
class AgentNetwork:
    agentSet, graphSet = [], []
    data1, data2, data3,iter = [], [], [], [] #Data points for SGP, PSG, and HDSG plot(x-axis is iterations, y-axis is function value)
    def __init__(self, agents):
        self.agentSet = agents #vertex set
    
    #Initialize edge set, in-neighbor sets, out-neighbor sets
    def initializeSets(self):
      self.iter.append(0)
      self.data1.append(1)
      self.data2.append(1)
      self.data3.append(1)
      if len(self.agentSet) > 0:
       for i in self.agentSet:
          for j in self.agentSet:
            self.graphSet.append((i,j))
            i.addToInSet(j)
            i.addToOutSet(j)
      else:
         raise IndexError('Must have at least two agents.')
    
    def addToAgentSet(self,x):
       self.agentSet.append(x)

    #Positive, nonincreasing, square summable but not summable step size choice
    def calculateStepSize(self,t):
       return 1/(t ** 2) 
       
    #Subgradient-Push Algorithm
    def SGP(self, iterations):
      for i in range(1,iterations + 1):
         self.iter.append(i)
         for j in self.agentSet:
            sumX,sumY = 0, 0
            for k in j.OSet:
               weight = 1/k.getOutSetSize()
               k.updateGradient(k.getX() / k.getY())
               sumX += weight * (k.getX() - self.calculateStepSize(i) * k.getGradient())
               sumY += weight * k.getY()
            j.setX(sumX)
            j.setY(sumY)
         self.data1.append(self.costFunction(j.getX()))
      
    #Push-Subgradient Algorithm
    def PSG(self,iterations):
      for i in range(1,iterations + 1):
         for j in self.agentSet:
            sumX,sumY = 0,0
            for k in j.ISet:
               weight = 1/k.getOutSetSize()
               sumX += (weight * k.getX()) - (self.calculateStepSize(i) * j.getGradient())
               sumY += weight * k.getY()
               j.updateGradient(j.getX() / j.getY()) 
            j.setX(sumX)
            j.setY(sumY)
         self.data2.append(self.costFunction(j.getX()))

    #Heterogeneous Distributed Subgradient Algorithm
    def HDSG(self,iterations):
       for i in range(1,iterations + 1):
          for j in self.agentSet:
             sumX,sumY = 0, 0
             switch1 = j.determineSwitch()
             for k in j.OSet:
               weight = 1/k.getOutSetSize()
               switch2 = k.determineSwitch()
               k.updateGradient(k.getX() /k.getY())
               sumX += weight * (k.getX() - (self.calculateStepSize(i) * k.getGradient() * switch2))
               - (self.calculateStepSize(i) * j.getGradient() * (1 - switch1))
               sumY += weight * k.getGradient()
               j.updateGradient(j.getX() / j.getY())
               k.updateGradient(k.getX() /k.getY())
             j.setX(sumX)
             j.setY(sumY)
          self.data3.append(self.costFunction(j.getX()))
     
     #Calculate global cost function
    def costFunction(self,x):
      sum = 0
      for a in self.agentSet:
         sum += a.getFunction()(x)
      return sum / len(self.agentSet)
    
#Agent with private cost function
class Agent():
    ISet, OSet, switch, x_current, y_current = [],[],0,1,1
    grad = 0
    def __init__(self,func):
        self.func = func
        self.updateGradient(self.x_current/self.y_current)
 
    #Returns the agent's cost function
    def getFunction(self):
        return self.func
    
    def useFunction(self,x):
       return self.func(x)
    
    #Add agent to network 
    def addAgent(Agent):
        AgentNetwork.agentSet.append(Agent)

    #Add agent to in-neighbor set
    def addToInSet(self,a):
       self.ISet.append(a)

    #Add agent to out-neighbor set
    def addToOutSet(self,a):
       self.OSet.append(a)

    #Returns number of in-neighbors
    def getInSetSize(self):
       return len(self.ISet)
    
    #Returns number of out-neighbors
    def getOutSetSize(self):
       return len(self.OSet)

    def setX(self,k):
       self.x_current = k
    
    def setY(self,k):
       self.y_current = k
    
    def getX(self):
       return self.x_current
    
    def getY(self):
       return self.y_current
    
    def updateGradient(self,x):
       self.grad = (self.useFunction(x + 0.00005) - self.useFunction(x))/(0.00005)

    def getGradient(self):
       return self.grad
    
    #Determines switch signal randomly in {0,1}
    def determineSwitch(self):
       randNum = random.randint(1,100)
       if randNum <= 50:
          Agent.switch = 0
       else:
          Agent.switch = 1
       return Agent.switch

agent1 = Agent(np.poly1d([1,0]))
agent2 = Agent(np.poly1d([1,0,0]))
agent3 = Agent(np.poly1d([1,0,1]))
agent4 = Agent(np.poly1d([2,5,0]))
agent5 = Agent(np.poly1d([5,-7]))
tNetwork = AgentNetwork([agent1,agent2,agent3,agent4,agent5])
tNetwork.initializeSets()   
tNetwork.SGP(100)
tNetwork.PSG(100)    
tNetwork.HDSG(100) 
plt.title('SGP,PSG,HDSG Plot')
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.ylim((-10,10))
plt.plot(tNetwork.iter,tNetwork.data1,'red',label='Subgradient-Push')
plt.plot(tNetwork.iter,tNetwork.data2,'green',label='Push-Subgradient')
plt.plot(tNetwork.iter,tNetwork.data3,'blue',label='Heterogeneous Subgradient')
plt.legend(loc="upper right")
plt.show()
      




