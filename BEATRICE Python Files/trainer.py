#torch is necessary for the neural network; .optim is for optimization
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
#shutil is for moving files; helps copy only the comments, etc.
import shutil
#os is for moving files around as well (??); doesn't throw an error if a value doesn't exist and is trying to be retrieved
import os
#glob is also for files; helps find file directory and path names
import glob
#all scripts packages are packages made specifically for BEATRICE 
from scripts.convert_to_gpu import gpu
from scripts.convert_to_gpu_and_tensor import gpu_t
from scripts.convert_to_gpu_scalar import gpu_ts
from scripts.convert_to_cpu import cpu
import scripts.generate_credible_sets as gen_cred
#plotting purposes
import matplotlib.pyplot as plt
#pickle serializes and deserializes python objects (writes and reads binary files)
import pickle
# few differences from torch.nn
import torch.nn.functional as F
#package is no longer needed; it's depricated. torch.tensor does the same thing
from torch.autograd import Variable
import pandas as pd
#used to produce images
import imageio
#plotting package
import seaborn as sn
#progress bar
from tqdm.auto import tqdm
#code timer
import time

#used to call matplotlib.pyplot later on
matplotlib.use('Agg')


#produces a file???
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 4)

#loads in file
def load_object(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        obj = pickle.load(input)
    return obj

#initializes neural network
class network(nn.Module):
    #K is the number of iterations for the NN, f_dim is feature dimension size, A is the ajacentcy matrix
    def __init__(self,K,f_dim,n_l,A,x):
        """Initialization of the neural network.
        """
        #super() used to call a super class
        super(network, self).__init__()
        
          #converts outputs to probabilities that sum up to 1
        self.softmax  = nn.Softmax(dim=1)
        #imp is the only one that is used
        self.imp      =[]
        #activation functions; activation functions are applied to a neuron's input to produce the output; multiple activation functions are used to prevent the vanishing gradient problem (producing the same output every time) and preventing linearity. ReLU is the only one used for the nn
        self.sig = nn.Sigmoid()
        #ReLU is mentioned in the supplemental materials page 4; thats the activation function that they specifically refer to in the picture of the neural network. it is the most common activation function used because it avoids the vanishing gradient problem
        self.rel = nn.ReLU()
        self.tan = nn.Tanh()

        #creates linear neural network layer; done so so that the nn can do computation and deal with matrices
        self.L1 = nn.Linear(f_dim[0], f_dim[1], bias=False)
        #normalizing layer to produce mean = 0 and sd = 1
        self.N1 = nn.LayerNorm(K)
        #2-step neural network that does the activation function and linear matrix activation
        self.A1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(f_dim[1], f_dim[2], bias=False)
            )
        self.N2 = nn.LayerNorm(K)
        self.A2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(f_dim[2], f_dim[3], bias=False)
            )
        self.N3 =  nn.LayerNorm(K)
        
        self.A3 = nn.Sequential(
            nn.ReLU()           
            )
        
        #final projection layer for binary output
        self.conc = nn.Linear(f_dim[n_l], 2,bias=False)

        
        self.A = A
        self.x = x
        self.degree = []
        ##nn.parameter holds trainable weights and biases for the nn
        self.variance = nn.Parameter(torch.rand(1))
        
        #makes it so that the equation is differentiable so that we can do math :)
    def gumbel(self,alpha,t):
        """ Generate Binary Concrete Vectors."""
        u = (-torch.log(-torch.log(gpu(torch.rand(alpha.size())))) + alpha)/t
        return F.softmax(u,dim=1)
    
        #works through the entire neural network and gives us new parameters
    def forward(self, T, samples):
        """ The inference module which generates the parameters of the binary 
        concrete distribution and generate samples of binary concrete vectors.      
        """
        #maybe the same epsilon that describes noncausal variants with nonzero effects (gpu_ts changes it to a readable form?)
        eps = gpu_ts(10**-7)
        X   = self.x.unsqueeze(1)        
        out = self.conc(self.A3(self.N3(self.A2(self.N2(self.A1(self.N1(self.L1(X).T).T).T).T).T).T)  )

        #applies exponential function to importance probabilities and makes that output readable
        imp     = gpu(torch.exp(out))
        imp_o   = imp[:,1]/torch.sum(imp,dim=1)
        self.imp = cpu(imp_o.detach()).data.numpy()
        
        eps = gpu_ts(10**-6)
        if self.training:
                 z_N     = self.gumbel(torch.log(imp.repeat(samples, 1)+eps), T) 
                 z_N1    = self.gumbel(torch.log(imp.repeat(1, 1)+eps), T) 
                 z_N2    = self.gumbel(torch.log(imp.repeat(1, 1)+eps), T)
                 if torch.isnan(torch.max(z_N)):
                     print(torch.max(z_N))

                 bin_concrete =  z_N[:,1].reshape(samples,len(imp_o))
                 bin_concrete1 = z_N1[:,1].reshape(1,len(imp_o))
                 bin_concrete2 = z_N2[:,1].reshape(1,len(imp_o))
        return bin_concrete,bin_concrete1,bin_concrete2, imp_o

          
        

class finemapper():
    #formatting stuff
    def __init__(self, model, opt, sch):
        self.model = model
        self.opt = opt
        self.scheduler = sch
    
    #function that approximates the bayes factor (whole purpose is to see how likely the observed z scores given that the causal traits are truly causal); epsilon is possibly error terms to prevent issues later on, ld = Linkage Disequilibrium of variants present in the locus, memo is the cache  of previously computed results, n_sub = number of subjects, sigma_sq = Variance of causal variants, cc = feature selection probabilities for the neural network, p0 = prior inclusion probabilities, k_c = maximum number of variables to consider, eps = threshold for deciding whether a variable is selected
    def abf(self, z, ld, memo, n_sub, sigma_sq, cc, p0, K_C, eps):

        #id_sort sorts the variables by importance
        id_sort = np.argsort(cpu(cc).data.numpy())[::-1]
        #shows the index for the top k_c variables
        id_sort = id_sort[:K_C]

        #sorted list of feature selection probabilities for the neural network(?)
        cc_t = cc[list(id_sort)]

        #combo of cc_t and ind keeps only the variables that are above a certain threshold
        ind = sorted(id_sort[cpu(torch.where(cc_t>eps)[0]).data.numpy()])
        #created a hashable key (numbers can be uniquely identified)
        ind_m  = tuple(ind)
        
        #assigns all feature selection probabilities to one to ensure that every single site has the same variance
        cc = gpu(torch.ones(len(z)))

        #first if ensures that there is at least one sight that exceeds our threshold epsilon
        if len(ind)>0:
            #if already computed, saves computation time
            if ind_m in memo:
                return memo[ind_m]

            #prior effect size scaled by sample size (assumes that all sites have the same variance)
            U =  n_sub*torch.diag(sigma_sq*cc)[:,ind]
            U = U[ind,:]

            #filters the ld matrix to only the rows of our selected sites
            V = LD[:,ind]
            V = V[ind,:]

            #filters the Z vector
            z = z[ind,:]

            #the log of the mean term in the likelihood
            log_mean = torch.logdet(torch.linalg.multi_dot([V, U, V]))

            #the sigma matrix of the likelihood inverted
            sigma_inv = torch.inverse(torch.linalg.multi_dot([V, U, V]))

            #the log of the exponential term in the likelihood
            log_exp = torch.linalg.multi_dot([z.T, sigma_inv, z])
        
            prior = 1 - p0
            prior[ind] = p0[ind]
        
            res =  -0.5 * (log_mean + log_exp) + torch.sum(torch.log(prior)) 
        
        
            memo[ind_m] = cpu(res).data.numpy()
        
            return res
        else:
            return
        
        
        
        
        
    def train(self, z_score, ld, temp, n_samples, sigma_sq, n_sub, p_0, num_iter,memo, epp, K_C, gamma):
        """ A training loop.
        Args:
            z_score: Normalized effect sizes obtained from GWAS summary statistics.
            ld: LD of variants present in the locus.
            temp: Temperature variable of the binary concrete distribution.
            n_sample: Number of monte carlo samples for integration.
            n_sub: Number of subjects.
            sigma_sq: Variance of causal variants.
            p_0: Prior probability of the underlying probability maps of binary concrete.
            num_iter: Number of times the training loop will run.
            memo: Dictionary that contains the probable causal configurations.
            epp: Epoch Number
            K_C: Sparsifying threshold.
            gamma: Threshold to creates reduced set of binary vectors.
        
        Output:
            total loss
            likelihood loss
            kl loss (regularization loss)
        """
        sigma_sq =  gpu_ts(sigma_sq)
        eps = gpu_ts(10**-7)
        ll_lik=[]
        ll_kl=[]
        ll_total=[]
        M = len(z_score)
        for n_b in range(num_iter):

            Z  =  Variable(z_score)            
            LD = Variable(ld)
            
            self.opt.zero_grad()
            # sets gradient for all parameters
            for param in self.model.parameters():
                param.requires_grad = True
            
            c,c1,c2, imp = self.model(temp,n_samples)
            loss = gpu_ts(0)
            loss_f = gpu_ts(0)
            Z = Z.unsqueeze(1)
            
            lik_loss = gpu_ts(0)
            kl_loss = gpu_ts(0)
            s2 = gpu_ts(0)
            for i in range(n_samples):
                (u,ind) = torch.topk(c[i],K_C)
                ind = ind[cpu(torch.where(u>0.01)[0]).data.numpy()]
                K_C = len(ind)
                if K_C ==0:
                    return [],[],[]
                cc = c[i]
                if epp>0:      
                    self.abf(Z, ld, memo, n_sub, sigma_sq, cc, p_0, K_C, gamma)
                
                U =  n_sub*torch.diag(sigma_sq*cc)[:,ind]
                U = U[ind,:]
                
                V = LD[:,ind]
                V = V[ind,:]
                
                Z = Z[ind,:]

                #the log of the mean term in the likelihood
                log_mean = torch.logdet(torch.linalg.multi_dot([V, U, V]))

                #the sigma matrix of the likelihood inverted
                sigma_inv = torch.inverse(torch.linalg.multi_dot([V, U, V]))

                #the log of the exponential term in the likelihood
                log_exp = torch.linalg.multi_dot([Z.T, sigma_inv, Z])

                log_likelihood = -0.5 * (log_mean + log_exp)
                
                lik_loss += -log_likelihood.squeeze()

                loss     += -log_likelihood.squeeze() 
                
                # Maybe sample seperately????????????
                
            # Analytic KL    
            x2 = imp[ind]
            x1 = p_0[ind]
            s1  = torch.sum(x2 * (torch.log(x2+eps) - torch.log(x1+eps)))
            s2 += torch.sum((1 - x2) * (torch.log(1 - x2+eps) - torch.log(1 - x1+eps))) + s1    
            kl_loss = s2
             
            loss_f = loss/n_samples  + kl_loss
            loss_f.backward()

            self.opt.step()
            ll_lik.append(cpu(lik_loss.detach()).data.numpy()/n_samples)
            ll_kl.append(cpu(kl_loss.detach()).data.numpy())
            ll_total.append(cpu(loss_f.detach()).data.numpy())
        
        return [np.mean(ll_total)], [np.mean(ll_lik)], [np.mean(ll_kl)]
            
 
           
        

##################################################################


def calculate_pip(M,bp):
    """Calculate posterior inclusion probabilities.
    """
    pip = np.zeros(bp)
    tot = 0
    for k in M:
        tot+= M[k]
        for i in k:
            pip[i]+= M[k]
            
    return np.squeeze(pip/tot)

def make_gif(M, bp, loc, store, ep):
    tr      = np.zeros(bp)
    tr[loc] = 1
    pip = []
    for k in M:   
        pip+= list(k)
    sn.histplot(pip, stat='count', bins=100,element='poly')
    #plt.stem(tr*(ep/2))
    plt.savefig(store+'/'+str(ep)+'.png')
    plt.close()
    
    list_of_files = filter( os.path.isfile,
                        glob.glob(store+ '/*.png') )
    # Sort list of files based on last modification time in ascending order
    filenames = sorted( list_of_files,
                        key = os.path.getmtime)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(store+'/movie.gif', images)
    return 
    
def regularize_ld(LD):
    """Regularize LD to make it non-singular
    """
    LD = (LD + LD.T)/2
    s, w = np.linalg.eig(cpu(LD).data.numpy())
    s = np.real(s)
    s_new = torch.zeros(len(s))
    if min(s)<10**-3:
        s_new = torch.ones(len(s))*(min(s)-10**-3)   
        print("\n Adding a constant {} to regularize LD".format(-min(s)+10**-3))
    LD = LD - gpu(torch.diag(s_new))    
    return LD
    
    
def reformat_memo(memo, p0):
    memo[tuple([])] = np.array([[torch.sum(torch.log(1-p0)).data.numpy()]])
    m0 = np.mean([val for val in memo.values()])
    for key in memo:
        memo[key] = min(10**15,np.exp(min(np.log(10**15),memo[key]-m0)))
    return m0
    
def main(options):    
    """
    options: A dictionary of hyper-parameters. 
    
    
    """
    
    start_time = time.time()
    ###################
    # Creat a folder to store figures.
    fig_location = os.path.join(options['target'],'figures')
    if os.path.exists(fig_location):
            shutil.rmtree(fig_location)
    os.mkdir(fig_location)
    ###################
    try:
        names = list(pd.read_table(options['z'],  sep=' ', header=None).to_numpy()[:,0])
        options['names'] = names
        Z  = gpu_t(pd.read_table(options['z'],  sep=' ', header=None).to_numpy()[:,1].astype(float))
        if torch.max(Z)==torch.inf:
            print('Z vector has inf as an element, converting it to 200')
            Z[torch.where(Z==torch.inf)[0]] = 200
            
        LD = gpu_t(pd.read_table(options['LD'], sep=' ', header=None).to_numpy())
    
    except BaseException as be:
        print(be)
        return
    if LD.size()[0]!=LD.size()[1]:
        return print('\n LD is not a square matrix')
    if LD.size()[0]!=Z.size()[0]:
        return print("\n Dimension of Z and dimension of LD are not same. Dim of Z = {}, dim of LD = {}".format(list(Z.size()), list(LD.size()) ))
    
    bp  = len(Z) # Number of variants..
    LD = regularize_ld(LD) 
    
    n_sub    = gpu_ts(options['n_sub'])
    
    if len(options['loc_true'])!=0:
        loc      = options['loc_true']
    else:
        loc = []
    

    ## Hyperparamters
    n_samples = options['MCMC_samples']
    sigma_sq =  options['sigma_sq']    
    n_epochs = options['max_iter']
    temp_lower_bound = gpu_ts(options['temp_lower_bound'])
    K_C = min(bp,options['sparsity_cl'])
    gamma_sp = options['gamma']
    num_iter = 1
    
    try:
        prior_loc = options['prior_location']
        p_0 = gpu_t(pd.read_table(prior_loc,  sep=' ', header=None).to_numpy()[:,1].astype(float))
        
    except:    
        p_0 = gpu_t(np.array([1/len(Z)]*len(Z)))

    # Initialize model.
    model = gpu(network(len(Z),[1]+options['NN'],3,LD,Z)   )

    # init optimizer
    opt_j       = optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), weight_decay=0)
    scheduler_j = torch.optim.lr_scheduler.StepLR(opt_j, step_size=1000, gamma = 0.5) # this will decrease the learning rate by factor of 0.1

    F_map = finemapper(model, opt_j, scheduler_j)
    F_map.S = torch.matmul(torch.inverse(LD),Z.unsqueeze(1))
    F_map.logdetLD = torch.logdet(LD)

    Loss = []
    Loss_lik=[]
    Loss_kl=[]
    memo ={}
    pip = np.zeros(len(Z))
    for n in tqdm(range(n_epochs+1)):
        temp  = torch.max(temp_lower_bound,gpu_ts(np.exp(-0.0001*n)))
        ll, ll_lik,ll_kl = F_map.train(Z, LD, temp, n_samples, sigma_sq,\
                                       n_sub, p_0, num_iter, memo, n, K_C, gamma_sp)
        F_map.scheduler.step()
        Loss.extend(ll)
        Loss_lik.extend(ll_lik)
        Loss_kl.extend(ll_kl)
        
            
        if n==n_epochs:  
            mean_memo = reformat_memo(memo, p_0)
            res_to_save={'loss':Loss,'lik_loss':Loss_lik,'kl_loss':Loss_kl, 'imp':F_map.model.imp,'loc':loc, 'pip':pip,'memo':memo, 'mean_memo':mean_memo}               
            pip = calculate_pip(memo, bp)  
            save_object(res_to_save, os.path.join(options['target'],'res'))
                
        
        if (n==(n_epochs//2) and n>0 and options['plot_loss'])or (n==n_epochs):
            real = np.zeros(len(Z))
            if len(options['loc_true'])!=0:
                real[loc] = 1
                plt.stem(real, linefmt='r-', markerfmt='ro')
            plt.stem(real, linefmt='r-', markerfmt='ro')
            plt.stem(pip)
            plt.xlabel('variants')
            plt.ylabel('PIP')
            plt.savefig(os.path.join(fig_location, 'pip.pdf'))
            plt.close()  

            plt.plot(Loss)
            plt.xlabel('epochs')
            plt.ylabel('Total Loss')
            plt.title('total loss')
            plt.savefig(os.path.join(fig_location,'total_loss.pdf'))
            plt.close()
            
            plt.plot(Loss_lik)
            plt.xlabel('epochs')
            plt.ylabel('Likelihood Loss')
            plt.title('lik loss')
            plt.savefig(os.path.join(fig_location,'lik_loss.pdf'))
            plt.close()
        
            plt.plot(Loss_kl)
            plt.title('kl loss')
            plt.xlabel('epochs')
            plt.ylabel('KL Regularization Loss')
            plt.savefig(os.path.join(fig_location,'kl_loss.pdf'))
            plt.close()

            real = np.zeros(len(Z))
            if len(options['loc_true'])!=0:
                real[loc] = 1
                plt.stem(real, linefmt='r-', markerfmt='ro')
            plt.stem(F_map.model.imp)
            plt.savefig(os.path.join(fig_location,'binary_concrete_prob.pdf'))
            plt.close()
            
            

    if options['get_cred']:
        gen_cred.main(options) 
    else:
        df = {'variant_index':list(range(bp)),'pip':pip, 'variant_names':names}
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(options['target'],'pip.csv'), index=False)
    
    
    finish_time = time.time()
    
    f = open(os.path.join(options['target'],'time'),'w')
    f.write(str(finish_time-start_time))
    f.close()    
    
    
    
    
    
    
    
    
    
