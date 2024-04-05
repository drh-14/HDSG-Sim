import numpy as np
import matplotlib.pyplot as plt
import random

#Network of Agents
class AgentNetwork:
    
    agentSet, graphSet= None
    
    def __init__(self):
        self.agentSet = [] #vertex set
        self.graphSet = [] #edge set 
    
    #Initialize edge set, in-neighbor sets, out-neighbor sets
    def initializeSets():
      if len(AgentNetwork.agentSet) > 1:
       for i in range(0,len(AgentNetwork.agentSet) - 1):
          for j in range(i+1,len(AgentNetwork.agentSet)):
            AgentNetwork.graphSet.append(AgentNetwork.agentSet[i],AgentNetwork.graphSet[j])
            AgentNetwork.agentSet[i].addToOutSet(AgentNetwork.agentSet[j])
            AgentNetwork.agentSet[j].addToInSet(AgentNetwork.agentSet[i])
      else:
         raise IndexError('Must have at least two agents.')

    #Positive, nonincreasing, square summable but not summable step size choice
    def stepsizeFunc(x):
       return 1/(x ** 2) 
       
    #Subgradient-Push Algorithm
    def SGP(AgentNetwork, stepsizeFunc, iterations):
      for i in range(0,iterations):
         sum_X,sum_Y = 0
         for j in range(0,len(AgentNetwork.agentSet)):
            weight = 1/AgentNetwork.agentSet[j].getOutSetSize()
            AgentNetwork.agentSet[j].updateGradient(AgentNetwork.agentSet[j].getX() / AgentNetwork.agentSet[j].getY())
            for k in range(0, AgentNetwork.agentSet[j].getOutSetSize()):
               sum_X = sum_X + weight * (AgentNetwork.agentSet[j].getX() -
                stepsizeFunc(iterations) * AgentNetwork.agentSet[j].updateGradient(AgentNetwork.agentSet[j].getX() /
               AgentNetwork.agentSet[j].getY()))
               sum_Y = sum_Y + AgentNetwork.agentSet[k].updateY(weight * AgentNetwork.agentSet[j].getY())
            AgentNetwork.agentSet[j].updateX(sum_X)
            AgentNetwork.agentSet[j].updateY(sum_Y)

    #Push-Subgradient Algorithm
    def PSG(AgentNetwork, stepsizeFunc, iterations):
      for i in range(0,iterations):
         sumX,sumY = 0
         for j in range(0,len(AgentNetwork.agentSet)):
            weight = 1/AgentNetwork.agentSet[j].getOutSetSize()
            for k in range(0,AgentNetwork.agentSet[j].getOutSetSize()):
               sumX = sumX + weight * AgentNetwork.agentSet[k].getX() - (stepsizeFunc(i + 1) * AgentNetwork.agentSet[k].getGradient())
               sumY = sumY + AgentNetwork.agentSet[k].updateY(weight * AgentNetwork.agentSet[j].getY())
               AgentNetwork.agentSet[k].updateGradient(AgentNetwork.agentSet[k].getX())
            
         AgentNetwork.agentSet[j].updateX(sumX)
         AgentNetwork.agentSet[j].updateY(sumY)

    #Heterogeneous Distributed Subgradient Algorithm
    def HDSG(AgentNetwork, iterations):
       for i in range(0,iterations):
          sum_X,sum_Y = 0
          for j in range(0,len(AgentNetwork.agentSet)):
             switchSignal = AgentNetwork.agentSet[j].determineSwitch()
             weight = 1/AgentNetwork.agentSet[j].getOutSetSize()
             for k in range(0, AgentNetwork.agentSet[j].getOutSetSize()):
               #Incomplete; placeholder
               sum_X = sum_X + weight * (AgentNetwork.agentSet[j].getX() - AgentNetwork.agentSet[j].stepsizeFunc(i) * AgentNetwork.agentSet[j].getSwitch())
     
     #Create global cost function
    def costFunction(x):
      return np.sum(x) / (len(x))
    
#Agent with private cost function
class Agent(AgentNetwork):
    __func, ISet, OSet, x_current, y_current, grad, switch = None
    def __init__(self,f):
        self.func = f
        self.ISet = [Agent]
        self.OSet = [Agent]

    #Returns the agent's cost function
    def getFunction():
        return Agent.__func
    
    #Add agent to network 
    def addAgent(Agent):
        AgentNetwork.agentSet.append(Agent)

    #Add agent to in-neighbor set
    def addToInSet(a):
       Agent.ISet.append(a)

    #Add agent to out-neighbor set
    def addToOutSet(a):
       Agent.OSet.append(a)

    #Returns number of in-neighbors
    def getInSetSize():
       return len(Agent.ISet)
    
    #Returns number of out-neighbors
    def getOutSetSize():
       return len(Agent.OSet)

    def updateX(k):
       Agent.x_current = k
       return k
    
    def updateY(k):
       Agent.y_current = k
       return k
    
    def getX():
       return Agent.x_current
    
    def getY():
       return Agent.y_current
    
    def updateGradient(x):
       Agent.grad = np.gradient([Agent.getFunc(x - 0.0001), Agent.getFunc(x-0.00005), Agent.getFunc(x),
       Agent.getFunc(x + 0.00005), Agent.getFunc(x + 0.0001)])
       return Agent.grad
    
    def getGradient():
       return Agent.grad
    
    #Determines switch signal randomly in {0,1}
    def determineSwitch():
       randNum = random.randint(1,100)
       if randNum <= 50:
          Agent.switch = 0
       else:
          Agent.switch = 1
       return Agent.switch

    def getSwitch():
       return Agent.switch
   


   
   
   
      
   
       

        







