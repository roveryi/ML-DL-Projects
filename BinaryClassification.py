# Different batch size can be tried to realize gradient descent, stochastic gradient descent and mini-batch gradient descent
batch_size = 64


from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms


## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
subset_indices = ((train_data.train_labels == 0) + (train_data.train_labels == 1)).nonzero().view(-1)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))


subset_indices = ((test_data.test_labels == 0) + (test_data.test_labels == 1)).nonzero().view(-1)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))

# Plot dataset
batch_idx, (example_data, example_targets) = next(enumerate(train_loader))
# print(example_data.shape)
# print(example_targets)
import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()

# Logistic regression class
class LogisticRegression(nn.Module): 
    def __init__(self, input_size): 
        super(LogisticRegression, self).__init__() 
        self.linear = nn.Linear(input_size,1) 
  
    def forward(self, x): 
#         out = torch.sigmoid(self.linear(x))
        out = self.linear(x)
        return out
    
class SupportVectorMachine(nn.Module):
    def __init__(self, input_size): 
        super(SupportVectorMachine, self).__init__() 
        self.linear = nn.Linear(input_size,1) 
  
    def forward(self, x): 
        out = self.linear(x) 
        return out


# Support vector machine class
class SupportVectorMachine(nn.Module): 
    def __init__(self, input_size): 
        super(SupportVectorMachine, self).__init__() 
        self.linear = nn.Linear(input_size,1)
  
    def forward(self, x): 
        out = self.linear(x)
        return out



#############################################################
# This part is using logistic regression to train the model #
#############################################################

# Training the Model
# Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.
num_epochs = 50
model = LogisticRegression(28*28)

criterion = nn.SoftMarginLoss(reduction='mean') 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 

for epoch in range(num_epochs):
    epoch_loss = 0
    num_violations = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        #Convert labels from 0,1 to -1,1
        labels = Variable(2*(labels.float()-0.5))
        
        labels = labels.view(labels.shape[0],1) # Reshape
        labels_pred = model(images) # Prediction
        loss = criterion(labels_pred,labels) # Compute Loss

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss
        
        prediction = torch.sign(labels_pred)
        
        total += labels.shape[0]
        num_violations += (prediction != labels).sum()
        
    print('Training loss after %d'%(epoch+1), 'th epoch: %f'%epoch_loss)
    print('Percentage of mislabeling after %d'%(epoch+1), 'th epoch: %f %%'%(100* num_violations.float()/total))
# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    ## Put your prediction code here
    prediction = model(images)

    prediction [prediction < 0] = 0
    prediction[prediction > 0 ] = 1
    
    correct += (prediction.view(-1).long() == labels).sum()
    total += images.shape[0]
print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))


# Alternatively, we can use logistic regression a little bit different way
# Cross entropy is used as loss function and labels are not needed to be transferred to -1 and 1
# Training the Model
# Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.
num_epochs = 50
model = LogisticRegression(28*28)

criterion = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 

for epoch in range(num_epochs):
    epoch_loss = 0
    num_violations = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        labels = labels.float()
        
        labels = labels.view(labels.shape[0],1) # Reshape
        labels_pred =torch.sigmoid(model(images))  # Prediction
        loss = criterion(labels_pred,labels) # Compute Loss

        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss
        
        labels_pred[labels_pred < 0.5] = 0
        labels_pred[labels_pred > 0.5] = 1

        total += labels.shape[0]
        num_violations += (labels_pred != labels).sum()
        
    print('Training loss after %d'%(epoch+1), 'th epoch: %f'%epoch_loss)
    print('Percentage of mislabeling after %d'%(epoch+1), 'th epoch: %f %%'%(100* num_violations.float()/total))

# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    ## Put your prediction code here
    prediction = torch.sigmoid(model(images))

    prediction [prediction < 0.5] = 0
    prediction[prediction > 0.5 ] = 1
    
    correct += (prediction.view(-1).long() == labels).sum()
    total += images.shape[0]
print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))


################################################################
# This part is using support vector machine to train the model #
################################################################
# Training the Model
# Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.
num_epochs = 50
model = SupportVectorMachine(28*28)

# criterion = nn.HingeEmbeddingLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001) 

for epoch in range(num_epochs):
    epoch_loss = 0
    num_violations = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        #Convert labels from 0,1 to -1,1
        labels = Variable(2*(labels.float()-0.5))
        
        ## TODD
        labels = labels.view(labels.shape[0],1)
        labels_pred = model(images)
        loss = torch.clamp(1 - labels.mul(model(images)),min = 0).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        epoch_loss += loss
        
        prediction = torch.sign(labels_pred)
        
        total += labels.shape[0]
        num_violations += (prediction != labels).sum()
        
    print('Training loss after %d'%(epoch+1), 'th epoch: %f'%epoch_loss)
    print('Percentage of mislabeling after %d'%(epoch+1), 'th epoch: %f %%'%(100* num_violations.float()/total))



# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    ## Put your prediction code here
    prediction = model(images)
    prediction[prediction > 0] = 1.
    prediction[prediction < 0] = 0.
    

    
    correct += (prediction.view(-1).long() == labels).sum()
    total += images.shape[0]
print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))
