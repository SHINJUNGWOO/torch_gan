
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import nn
import torch



class classifier_model(nn.Module):
    def __init__(self):
        super(classifier_model,self).__init__()
        self.conv1 = nn.Conv2d(1,32,5,padding=2)
        self.conv2 = nn.Conv2d(32,64,5,padding=2)
        self.conv3 = nn.Conv2d(64,64,5,padding=2)
        self.dense1 = nn.Linear(64 * 16 * 16, 1024)
        self.dense2 = nn.Linear(1024, 10)

        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x


data_loader = DataLoader(
    datasets.MNIST('data/mnist', train=True, download=True,
    transform=transforms.Compose([transforms.Resize((64, 64)),
    transforms.ToTensor()])),
    batch_size=128,
    shuffle=True
)

classifier = classifier_model()
classifier.train()
classifier.cuda()
optimizer = torch.optim.Adam(classifier.parameters(),lr=1e-3)
criterion = nn.CrossEntropyLoss()
running_loss = 0
for epoch in range(15):
    for i,(data, target) in enumerate(data_loader):
        data, target = Variable(data), Variable(target)
        data=data.cuda()
        target=target.cuda()
        optimizer.zero_grad()
        output = classifier(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 0:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            classifier.eval()
            correct = 0
            for data, target in data_loader:
                data, target = Variable(data), Variable(target)
                data = data.cuda()
                target =target.cuda()
                output = classifier(data)
                prediction = output.data.max(1)[1]
                correct += prediction.eq(target.data).sum()

            print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(data_loader.dataset)))
