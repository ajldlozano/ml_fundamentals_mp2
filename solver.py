from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
import pandas as pd

class PolySolverNet:
  def __init__(self):
    self.l1 = Tensor.uniform(4,1)
    
  def forward(self, x):
    print("x")
    print(x.data)
    print(self.l1.data)
    something=x.dot(self.l1)
    print(something.data)
    return something
    #return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

  def train_model(self,X,Y):
    epoch=200
    i=0
    opt = optim.SGD([self.l1], lr=0.001)
    opt.zero_grad()

    while i <epoch: #try epoch as a batch
      outs=[]

      for x,y in zip(X,Y):

        y = Tensor([y], requires_grad=True)
        x = Tensor(x, requires_grad=True)

        out = model.forward(x)

        #outs= Tensor(out,requires_grad=True)
        print("error comp")
        print(y.data)
        print(out.data)
        loss =((out-y)).abs()#y.sub(out).pow(2).mean()#(out-y).abs()#.mul(out-y).mean()#y.mul(out).mean()
        #print(loss.data)
        opt.zero_grad()        
        loss.backward()
        opt.step()
        print("current loss:")
        print(loss.data)
      i=i+1

    return loss




# ... and complete like pytorch, with (x,y) data


train = pd.read_csv('data_train.csv')
x = train['x']
#x=(x-x.min())/(x.max()-x.min())

#xstd=x.std()
x= (x-x.mean())/x.std()

x2 = train['x'] * train['x']
#x2=(x2-x2.min())/(x2.max()-x2.min())
#x2std=x2.std()
x2= (x2-x2.mean())/x2.std()

x3 = train['x'] * train['x'] * train['x']
#x3=(x3-x3.min())/(x3.max()-x3.min())
#x3=x3/x3.max()
#x3std=x3.std()
x3= (x3-x3.mean())/x3.std()

x4 = train['x'] * train['x'] * train['x'] * train['x']
#x4=x4/x4.max()
#x4=(x4-x4.min())/(x4.max()-x4.min())
#x4std=x4.std()
x4= (x4-x4.mean())/x4.std()

Y = train['y']
#Y=Y/Y.max()
#Y = (Y-Y.mean())/Y.std()


X= [list(x) for x in  zip(x,x2,x3,x4)] 
print("--------------------Y---------------")
print(Y)
print("--------------------X---------------")
print(X)




model = PolySolverNet()
loss= model.train_model(X,Y)
print(loss.data)
params=model.l1.data
print(params)

train = pd.read_csv('data_train.csv')
sx=train['x'].iloc[0]
sx2=sx*sx
sx3=sx2*sx
sx4=sx3*sx
for_scaling=model.forward(Tensor([sx,sx2,sx3,sx4], requires_grad=True))
print("scale correction")
print((for_scaling.data/train['y'].iloc[0]))
scaled_params=params/(for_scaling.data/train['y'].iloc[0])
print(scaled_params)

test = pd.read_csv('data_test.csv')
tx=test['x']
tx2=tx*tx
tx3=tx*tx*tx
tx4=tx*tx*tx*tx
ty=test['y']
for t in zip(tx,tx2,tx3,tx4,ty):
  print("input: ", t[0])
  print("correct: ")
  print(t[-1])
  print("predicted: " )
  print(t[0]*scaled_params[0]+t[1]*scaled_params[1]+t[2]*scaled_params[2]+t[3]*scaled_params[3])



print("")
print("Final Coefficients: for x x2 x3 and x4")
print("for x:")
print(scaled_params[0])
print("for x^2:")
print(scaled_params[1])
print("for x^3:")
print(scaled_params[2])
print("for x^4:")
print(scaled_params[3])


