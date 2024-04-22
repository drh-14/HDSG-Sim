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
      if len(self.agentSet) > 1:
       for i in range(0,len(self.agentSet) - 1):
          for j in range(i,len(self.agentSet)):
            self.graphSet.append((self.agentSet[i],self.agentSet[j]))
            self.agentSet[i].addToOutSet(self.agentSet[j])
            self.agentSet[j].addToInSet(self.agentSet[i])
      else:
         raise IndexError('Must have at least two agents.')
    
    def addToAgentSet(self,x):
       self.agentSet.append(x)

    #Positive, nonincreasing, square summable but not summable step size choice
    def stepsizeFunc(self,x):
       if x == 0:
          return 0
       return 1/(x ** 2) 
       
    #Subgradient-Push Algorithm
    def SGP(self, iterations):
      for i in range(0,iterations):
         self.iter.append(i)
         sum_X,sum_Y = 0, 0
         for j in self.agentSet:
            if j.getOutSetSize() == 0:
               weight = 1
            else:
               weight = 1/j.getOutSetSize()
            j.updateGradient(j.getX() / j.getY())
            for k in j.OSet:
               sum_X = sum_X + weight * (j.getX() -
                self.stepsizeFunc(i) * j.updateGradient(j.getX() /
               j.getY()))
               sum_Y = sum_Y + k.updateY(weight * j.getY())
            j.updateX(sum_X)
            j.updateY(sum_Y)
         self.data1.append(self.costFunction(j.getX))
      

    #Push-Subgradient Algorithm
    def PSG(self,iterations):
      for i in range(0,iterations):
         self.iter.append(i)
         sumX,sumY = 0, 0
         for j in self.agentSet:
            j.updateGradient(j.getX() / j.getY())
            if j.getOutSetSize() == 0:
               weight = 1
            else:
               weight = 1/j.getOutSetSize()
            for k in j.OSet:
               k.updateGradient(k.getX() / k.getY())
               sumX = sumX + weight * k.getX() - (self.stepsizeFunc(i) * k.getGradient())
               sumY = sumY + k.updateY(weight * j.getY())
               k.updateGradient(k.getX() / k.getY()) 
            j.updateX(sumX)
            j.updateY(sumY)
         self.data2.append(self.costFunction(j.getX()))
      print(self.data2)
      
    #Heterogeneous Distributed Subgradient Algorithm
    def HDSG(self,iterations):
       for i in range(0,iterations):
          AgentNetwork.iter.append(i)
          sum_X,sum_Y = 0, 0
          for j in self.agentSet:
             j.updateGradient(j.getX() / j.getY())
             switchSignal = j.determineSwitch()
             weight = 1/j.getOutSetSize()
             for k in j.OSet:
               k.updateGradient(k.getX() /k.getY())
               sum_X = sum_X + weight * (k.getX() - k.getFunction() * 
               k.getGradient() * switchSignal) - AgentNetwork.agentSet[k] * j.getGradient()
               (1 - switchSignal)
               sum_Y = sum_Y + weight * k.getY()
     
     #Calculate global cost function
    def costFunction(self,x):
      sum = 0
      for a in self.agentSet:
         f = a.getFunction()
         sum = sum + f(x)
      return sum / len(self.agentSet)
    
#Agent with private cost function
class Agent():
    ISet, OSet, switch = None,None,[]
    x_current, y_current, grad = 1,1,0
    def __init__(self,func):
        self.func = func
        self.ISet = []
        self.OSet = []
 
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

    def updateX(self,k):
       self.x_current = k
       return k
    
    def updateY(self,k):
       self.y_current = k
       return self.y_current
    
    def getX(self):
       return self.x_current
    
    def getY(self):
       return self.y_current
    
    def updateGradient(self,x):
       self.grad = (self.useFunction(x) - self.useFunction(x + 0.00000025)) / 0.00000025
    
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

agent1 = Agent(np.sqrt)
agent2 = Agent(np.sqrt)
agent3 = Agent(np.square)
agent4 = Agent(np.cbrt)
agent5 = Agent(np.log2)
a = AgentNetwork([agent1, agent2, agent3, agent4, agent5])
a.initializeSets()
a.PSG(5)

   
   
   
      
   
       

        







