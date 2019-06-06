import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
from torchvision import transforms
from torchsummary import summary
from torchvision import datasets, models, transforms
import os
import torch.optim as optim
import copy
from torch.optim import lr_scheduler

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),stride=1,padding=0,dilation=1,groups=1,bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SeparableBlock(nn.Module):
    def __init__(self,in_channels,out_channals,SepKsize):
        super(SeparableBlock,self).__init__()

        self.relu = nn.ReLU()
        self.sepConv = SeparableConv2d(in_channels,out_channals,SepKsize,1,1,1,False)
        self.bn = nn.BatchNorm2d(out_channals)

    def forward(self,x):
        x = self.relu(x)
        x = self.sepConv(x)
        x = self.bn(x)
        return x

class MidleBlock(nn.Module):
    def __init__(self,in_channels,ksize):
        super(MidleBlock,self).__init__()

        self.blockSep1 = SeparableBlock(in_channels,in_channels,ksize)
        self.blockSep2 = SeparableBlock(in_channels,in_channels,ksize)
        self.blockSep3 = SeparableBlock(in_channels,in_channels,ksize)

    def forward(self,x):
        x = self.blockSep1(x)
        x = self.blockSep2(x)
        x = self.blockSep3(x)
        return x


class EntryFlow(nn.Module):
    def __init__(self,size_list,kernel_size,stride):
        super(EntryFlow,self).__init__()

        self.convIn1 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bnIn1 = nn.BatchNorm2d(32)
        self.reluIn1 = nn.ReLU()

        self.convIn2 = nn.Conv2d(32,64,3,bias=False)
        self.bnIn2 = nn.BatchNorm2d(64)
        self.reluIn2 = nn.ReLU()

        self.conv1 = nn.Conv2d(size_list[0][0],size_list[0][1],1,stride)
        self.sep1 = SeparableConv2d(size_list[0][0],size_list[0][1],kernel_size,1,1,1,False)
        self.block1 = SeparableBlock(size_list[0][1],size_list[0][1],kernel_size)
        self.maxpooling1 = nn.MaxPool2d(kernel_size,stride,1)
        
        self.conv2 = nn.Conv2d(size_list[1][0],size_list[1][1],1,stride)
        self.block2 = SeparableBlock(size_list[1][0],size_list[1][1],kernel_size)
        self.block3 = SeparableBlock(size_list[1][1],size_list[1][1],kernel_size)
        self.maxpooling2 = nn.MaxPool2d(kernel_size,stride,1)

        self.conv3 = nn.Conv2d(size_list[2][0],size_list[2][1],1,stride)
        self.block4 = SeparableBlock(size_list[2][0],size_list[2][1],kernel_size)
        self.block5 = SeparableBlock(size_list[2][1],size_list[2][1],kernel_size)
        self.maxpooling3 = nn.MaxPool2d(kernel_size,stride,1)

    def forward(self,input):

        input = self.convIn1(input)
        input = self.bnIn1(input)
        input = self.reluIn1(input)

        input = self.convIn2(input)
        input = self.bnIn2(input)
        input = self.reluIn2(input)


        resInput = self.conv1(input)
        input = self.sep1(input)
        input = self.block1(input)
        input = self.maxpooling1(input)
        input = input + resInput

        resInput = self.conv2(input)
        input = self.block2(input)
        input = self.block3(input)
        input = self.maxpooling2(input)
        input = input + resInput

        resInput = self.conv3(input)
        input = self.block4(input)
        input = self.block5(input)
        input = self.maxpooling3(input)
        out = input + resInput

        return out

class MiddleFlow(nn.Module):
    def __init__(self,in_channels,ksize):
        super(MiddleFlow,self).__init__()

        self.midleBlock1 = MidleBlock(in_channels,ksize)
        self.midleBlock2 = MidleBlock(in_channels,ksize)
        self.midleBlock3 = MidleBlock(in_channels,ksize)
        self.midleBlock4 = MidleBlock(in_channels,ksize)
        self.midleBlock5 = MidleBlock(in_channels,ksize)
        self.midleBlock6 = MidleBlock(in_channels,ksize)
        self.midleBlock7 = MidleBlock(in_channels,ksize)
        self.midleBlock8 = MidleBlock(in_channels,ksize)

    def forward(self,x):
        x = x + self.midleBlock1(x)
        x = x + self.midleBlock2(x)
        x = x + self.midleBlock3(x)
        x = x + self.midleBlock4(x)
        x = x + self.midleBlock5(x)
        x = x + self.midleBlock6(x)
        x = x + self.midleBlock7(x)
        x = x + self.midleBlock8(x)
        return x
        
class ExitFlow(nn.Module):
    def __init__(self,szList,kernel_size,stride):
        super(ExitFlow,self).__init__()
        
        self.convRes1 = nn.Conv2d(szList[0][0],szList[1][1],1,stride,0,1,1,False)
        self.block1 = SeparableBlock(szList[0][0],szList[0][1],kernel_size)
        self.block2 = SeparableBlock(szList[1][0],szList[1][1],kernel_size)
        self.maxpooling1 = nn.MaxPool2d(kernel_size,stride=stride,padding=1)

        self.sepconvF1 = SeparableConv2d(szList[2][0],szList[2][1],kernel_size,1,1)
        self.bnF1 = nn.BatchNorm2d(szList[2][1])
        self.reluF1 = nn.ReLU()

        self.sepconvF2 = SeparableConv2d(szList[3][0],szList[3][1],kernel_size,1,1)
        self.bnF2 = nn.BatchNorm2d(szList[3][1])
        self.reluF2 = nn.ReLU()

        #self.gap=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input):
        resInput = self.convRes1(input)

        input = self.block1(input)
        input = self.block2(input)
        input = self.maxpooling1(input)
        input = resInput + input

        input = self.sepconvF1(input)
        input = self.bnF1(input)
        input = self.reluF1(input)

        input = self.sepconvF2(input)
        input = self.bnF2(input)
        input = self.reluF2(input)

        input = torch.mean(input.view(input.size(0),input.size(1), -1), dim=2)

        return input


class Xception(nn.Module):
    def __init__(self,num_output):
        super(Xception,self).__init__()

        EntryFlowSzList = [[64,128],
                          [128,256],
                          [256,728]]
        ExitFlowSzList = [[728,728],
                        [728,1024],
                        [1024,1536],
                        [1536,2048]]
        self.entry = EntryFlow(EntryFlowSzList,(3,3),(2,2))
        self.middle = MiddleFlow(728,(3,3))
        self.exit = ExitFlow(ExitFlowSzList,(3,3),(2,2))    
        self.dence = nn.Linear(ExitFlowSzList[3][1],num_output)
    def forward(self,x):
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = self.dence(x)
        return x


def xception():

    model = Xception()
    model = model.float().cuda()
    model.eval()
    #example = torch.rand(1, 1, 250, 200).cuda()
    out = model(example)
    print(out.shape)
    summary(model,(1, 200, 150))


#xception()
def train():

    data_transforms = {
        "train":transforms.Compose([transforms.Resize((250,200)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(degrees=(-2,2),center=(26,64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        "val":transforms.Compose([transforms.Resize((250,200)),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
            #transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            #transforms.Normalize([127.5,127.5,127.5],[127.5,127.5,127.5])
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        }
    data_dir = "D:\\tasks\\dogs-vs-cats"
    saveModelPath = "Xceptioncatdog.pt"
    #param
    batch_size = 32
    num_epochs = 40
    phases = ["train","val"]

    imageFolders = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in phases}

    dataLoader = {x:torch.utils.data.DataLoader(imageFolders[x],batch_size=batch_size,shuffle=True,num_workers=4) for x in phases}

    
    dataset_sizes = {x:len(imageFolders[x]) for x in phases}
    class_names = imageFolders["train"].classes
    print(dataset_sizes)
    print(class_names)
    print("count classes: ",len(class_names))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Xception(2).to(device)
    print("Creeat model")
    best_model_wts = copy.deepcopy(model.state_dict())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.045,momentum=0.9,weight_decay=0.0005)
    scheduler=lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.94)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1,num_epochs))

        for phase in phases:

            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0.0
            count_iter = 0

            for inputs,labels in dataLoader[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                count_iter+=1
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss+=loss.item() * inputs.size(0)
                running_corrects+=torch.sum(preds == labels.data)
                #print("iteration: ",count_iter," loss: ",loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase,epoch_loss,epoch_acc))
           
            if phase == "val" and epoch_acc > best_acc:
                    #pth=pathCheckpoint+"chpt_"+str(epoch_acc.item())+"_.pt"
                    #torch.save(model,pth)
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)


    #trainedModel=trainedModel.float()
    model.eval()
    example = torch.rand(1, 3, 250, 200).to(device)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(saveModelPath)

if __name__ == "__main__":
    train()