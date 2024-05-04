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
               sumX += weight * (j.getX() - self.calculateStepSize(i) * k.getGradient())
               sumY += weight * k.getY()
               k.updateGradient(j.getX() / j.getY())
            j.setX(sumX)
            j.setY(sumY)
         self.data1.append(self.costFunction(j.getX()))
      print(self.data1)
      
    #Push-Subgradient Algorithm
    def PSG(self,iterations):
      for i in range(1,iterations + 1):
         for j in self.agentSet:
            sumX,sumY = 0,0
            for k in j.ISet:
               weight = 1/k.getOutSetSize()
               sumX += (weight * k.getX()) - (self.calculateStepSize(i) * j.getGradient())
               sumY += weight * k.getY()
            j.setX(sumX)
            j.setY(sumY)
            j.updateGradient(j.getX() / j.getY()) 
         self.data2.append(self.costFunction(j.getX()))
      print(self.data2)

    #Heterogeneous Distributed Subgradient Algorithm
    def HDSG(self,iterations):
       for i in range(1,iterations + 1):
          for j in self.agentSet:
             sumX,sumY = 0, 0
             j.updateGradient(j.getX() / j.getY())
             switchSignal = j.determineSwitch()
             weight = 1/j.getOutSetSize()
             for k in j.OSet:
               k.updateGradient(k.getX() /k.getY())
               sumX += weight * (k.getX() - self.calculateStepSize(i) * k.getGradient() * switchSignal)
               - (self.calculateStepSize(i) * j.getGradient() * (1 - switchSignal))
               sumY += weight * k.getGradient()
          j.setX(sumX)
          j.setY(sumY)
          self.data3.append(j.getX())
     
     #Calculate global cost function
    def costFunction(self,x):
      sum = 0
      for a in self.agentSet:
         sum += a.getFunction()(x)
      return sum / len(self.agentSet)
    
#Agent with private cost function
class Agent():
    ISet, OSet, switch, x_current, y_current = [],[],0,5,1
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

    def getSwitch():
       return Agent.switch

agent1 = Agent(np.square)
agent2 = Agent(np.cos)
agent3 = Agent(np.sinc)
agent4 = Agent(np.sin)
tNetwork = AgentNetwork([agent1,agent2,agent3,agent4])
tNetwork.initializeSets()   
tNetwork.SGP(50)
tNetwork.PSG(50)
tNetwork.HDSG(50)
plt.plot(tNetwork.iter,tNetwork.data1, 'red')
plt.plot(tNetwork.iter,tNetwork.data2, 'green')
plt.plot(tNetwork.iter,tNetwork.data3, 'blue')

plt.show()
      
        







