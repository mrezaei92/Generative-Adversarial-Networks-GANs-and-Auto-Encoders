import model
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from keras.datasets import mnist
from sklearn.manifold import TSNE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# build the model
model=AVE(784,784).double().to(device)
encoder=model.encoder
decoder=model.decoder
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
bath_size=16
num_batch=np.int16(np.floor(x_train.shape[0]/16))
MSE_loss=nn.MSELoss()

## train

model.train()
B=1
for epoch in range(10):  # loop over the dataset multiple times
        t=np.random.permutation(x_train.shape[0])
        x_train=x_train[t]
        for j in range(num_batch):
            inputs=x_train[(j*bath_size):((j+1)*bath_size),:,:]
            inputs=B*torch.from_numpy(inputs.reshape(bath_size,-1)).double()#.to(device)
            labels = inputs

            optimizer.zero_grad()
            output,mu,sigma,_=model(inputs)
            # forward + backward + optimize
            #Reconstruction_loss = F.binary_cross_entropy(output, labels, reduction='sum') / bath_size
            Reconstruction_loss=MSE_loss(output,labels)
            KL_divergence = 0.5 * torch.sum( torch.pow(mu, 2) + torch.pow(sigma, 2) - torch.log(1e-8 + torch.pow(sigma, 2)) - 1 ).sum() / bath_size
            loss = Reconstruction_loss+KL_divergence

            loss.backward()
            optimizer.step()
            
            if j%500==0:
                print("Loss= {} after batch iteration= {}".format(loss.item(),j))
                      
        print("End of epoch: ",epoch)
        #print(np.mean(running_loss),time.time() - start_time_iter)
        print("Starting next epoch")

print('Finished Training')

# build the encoded version of the data
z_vectors=[]
for i in range(x_test.shape[0]):
    img=x_test[i]
    img_tensor=torch.from_numpy(img.reshape(1,-1)).double()
    reconstructed,m,s,z=model(img_tensor)
    z_vectors.append(z.detach().numpy())
    
z_vectors=np.squeeze(np.array(z_vectors))

## Use T-SNE to transform the z vectors to a 2-dimensional space for visualization purposes
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(z_vectors)

# Visualize
colors=["red","black","blue","orange","pink","gray","yellow","brown","green","tan"]
for i in range(10):
    sub_set=y_test==i
    plt.scatter(tsne_results[sub_set,0],tsne_results[sub_set,1],c=colors[i],label="Digit {}".format(i))

plt.legend(loc='upper center',bbox_to_anchor=(0, 1.05))


