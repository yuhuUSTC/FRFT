import torch
import torch.nn as nn
import math


# core module
class FRFT(nn.Module):
    def __init__(self, in_channels, order=0.5):
        super(FRFT, self).__init__()
        C0 = int(in_channels/3)
        C1 = int(in_channels) - 2*C0
        self.conv_0 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)
        self.conv_05 = nn.Conv2d(2*C1, 2*C1, kernel_size=1, padding=0)
        self.conv_1 = nn.Conv2d(2*C0, 2*C0, kernel_size=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.order = nn.Parameter(torch.randn(1))



    def dfrtmtrx(self, N, a):
        # Approximation order
        app_ord = 2
        Evec = self.dis_s(N,app_ord).cuda()
        Evec = Evec.to(dtype=torch.complex64)
        even = 1 - (N%2)
        l = torch.tensor(list(range(0,N-1)) + [N-1+even]).cuda()
        f = torch.diag(torch.exp(-1j*math.pi/2*a*l))
        F = N**(1/2)*torch.einsum("ij,jk,ni->nk", f, Evec.T, Evec)
        return F

    def dis_s(self, N, app_ord):  
        app_ord = int(app_ord / 2) 
        s = torch.cat((torch.tensor([0, 1]), torch.zeros(N-1-2*app_ord), torch.tensor([1])))
        S = self.cconvm(N,s) + torch.diag((torch.fft.fft(s)).real);

        p = N
        r = math.floor(N/2)
        P = torch.zeros((p,p))
        P[0,0] = 1
        even = 1 - (p%2)
        
        for i in range(1,r-even+1):
            P[i,i] = 1/(2**(1/2))
            P[i,p-i] = 1/(2**(1/2))
            
        if even:
            P[r,r] = 1
            
        for i in range(r+1,p):
            P[i,i] = -1/(2**(1/2))
            P[i,p-i] = 1/(2**(1/2))

        CS = torch.einsum("ij,jk,ni->nk", S, P.T, P)
        C2 = CS[0:math.floor(N/2+1), 0:math.floor(N/2+1)]
        S2 = CS[math.floor(N/2+1):N, math.floor(N/2+1):N]
        ec, vc = torch.linalg.eig(C2)
        es, vs = torch.linalg.eig(S2)
        ec = ec.real
        vc = vc.real
        es = es.real
        vs = vs.real
        qvc = torch.vstack((vc, torch.zeros([math.ceil(N/2-1), math.floor(N/2+1)])))
        SC2 = P@qvc # Even Eigenvector of S
        qvs = torch.vstack((torch.zeros([math.floor(N/2+1), math.ceil(N/2-1)]),vs))
        SS2 = P@qvs # Odd Eigenvector of S
        idx = torch.argsort(-ec)
        SC2 = SC2[:,idx]
        idx = torch.argsort(-es)
        SS2 = SS2[:,idx]
        
        if N%2 == 0:
            S2C2 = torch.zeros([N,N+1])
            SS2 = torch.hstack([SS2, torch.zeros((SS2.shape[0],1))])
            S2C2[:,range(0,N+1,2)] = SC2;
            S2C2[:,range(1,N,2)] = SS2
            S2C2 = S2C2[:, torch.arange(S2C2.size(1)) != N-1]
        else:
            S2C2 = torch.zeros([N,N])
            S2C2[:,range(0,N+1,2)] = SC2;
            S2C2[:,range(1,N,2)] = SS2
        
        return S2C2 

    def cconvm(self, N, s):
        M = torch.zeros((N,N))
        dum = s
        for i in range(N):
            M[:,i] = dum
            dum = torch.roll(dum,1)
        return M

    def FRFT2D(self, matrix):
        N, C, H, W = matrix.shape
        h_test = self.dfrtmtrx(H, self.order).cuda()
        w_test = self.dfrtmtrx(W, self.order).cuda()
        h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=C, dim=0)
        h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=N, dim=0)
        w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=C, dim=0)
        w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=N, dim=0)

        out = []
        matrix = torch.fft.fftshift(matrix, dim=(2, 3)).to(dtype=torch.complex64)

        out = torch.matmul(h_test, matrix)
        out = torch.matmul(out, w_test)

        out = torch.fft.fftshift(out, dim=(2, 3))
        return out


    def IFRFT2D(self, matrix):
        N, C, H, W = matrix.shape
        h_test = self.dfrtmtrx(H, -self.order).cuda()
        w_test = self.dfrtmtrx(W, -self.order).cuda()
        h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=C, dim=0)
        h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=N, dim=0)
        w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=C, dim=0)
        w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=N, dim=0)

        out = []
        matrix = torch.fft.fftshift(matrix, dim=(2, 3)).to(dtype=torch.complex64)
        
        out = torch.matmul(h_test, matrix)
        out = torch.matmul(out, w_test)

        out = torch.fft.fftshift(out, dim=(2, 3))
        return out


    def forward(self, x):
        N, C, H, W = x.shape

        C0 = int(C/3)
        x_0 = x[:, 0:C0, :, :]
        x_05 = x[:, C0:C-C0, :, :]
        x_1 = x[:, C-C0:C, :, :]

        # order = 0
        x_0 = self.conv_0(x_0)
        
        # order = 0.5
        Fre = self.FRFT2D(x_05)
        Real = Fre.real
        Imag = Fre.imag
        Mix = torch.concat((Real, Imag), dim=1)
        Mix = self.conv_05(Mix)
        Real1, Imag1 = torch.chunk(Mix, 2, 1)
        Fre_out = torch.complex(Real1, Imag1)
        IFRFT = self.IFRFT2D(Fre_out)
        IFRFT = torch.abs(IFRFT)/(H*W)

        # order = 1
        fre = torch.fft.rfft2(x_1, norm='backward')
        real = fre.real
        imag = fre.imag
        mix = torch.concat((real, imag), dim=1)
        mix = self.conv_1(mix)
        real1, imag1 = torch.chunk(mix, 2, 1)
        fre_out = torch.complex(real1, imag1)
        x_1 = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        output = torch.cat([x_0, IFRFT, x_1], dim=1)
        output = self.conv2(output)

        return output




#--------------------------------------------------------------------------------------
# your original network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.conv2(out1)
        return out


#--------------------------------------------------------------------------------------
# plug in our FRFT module and baseline module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)
        self.FRFT = FRFT(in_channels=C0, order=0.5)


    def forward(self, x):
        out1 = self.conv1(x)  # out1.shape: B, C, H, W
        out1 = self.FRFT(out1) 
        out = self.conv2(out1)
        return out
    

