import torch
import torch.nn as nn
from model import model_subnetwork1
from .layer_utils.resnet import ResnetBackbone
from .layer_utils.funcs import get_norm_layer
import model.blocks as blocks

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PDU(nn.Module):  # physical block
    def __init__(self, channel):
        super(PDU, self).__init__()
    
        self.ka = nn.Sequential(
            nn.Conv2d(64, channel, 3, padding=1, bias=True),nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1, bias=True),nn.Sigmoid()
        )
        
        self.td =  model_subnetwork1.DefaultModel()


    def forward(self, I_alpha, I, delta_I):

        A_infinity = self.ka(I_alpha)
        T = self.td( I_alpha, I, delta_I)
        j = torch.clamp((T * A_infinity ) / (A_infinity - I + T +1e-7), min=0, max=1)
        return j


class Group1(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group1, self).__init__()
        
        pre_precess1_0 = [
            conv(3, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True)]
        self.pre1_0 = nn.Sequential(*pre_precess1_0)
        pre_precess1_1 = [
            conv(3, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True)]
        self.pre1_1 = nn.Sequential(*pre_precess1_1)
        pre_precess1_2 = [
            conv(3, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True)]
        self.pre1_2 = nn.Sequential(*pre_precess1_2)
        pre_precess1 = [
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True)]
        self.pre1 = nn.Sequential(*pre_precess1)

        pre_precess2 = [
            conv(3, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size), nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size), nn.ReLU(inplace=True)]
        self.pre2 = nn.Sequential(*pre_precess2)
        pre_precess3 = [
            conv(3, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size), nn.ReLU(inplace=True)]
        self.pre3 = nn.Sequential(*pre_precess3)

        post_precess = [
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True),conv(dim, dim, kernel_size),nn.ReLU(inplace=True)]
        self.post_precess = nn.Sequential(*post_precess)

        self.pdu = PDU(dim)

    def forward(self, I_alpha, I, delta_I):

        f_I_alpha_0 = self.pre1_0(I_alpha[:,0:3,:,:])
        f_I_alpha_1 = self.pre1_1(I_alpha[:, 3:6, :, :])
        f_I_alpha_2 = self.pre1_2(I_alpha[:, 6:9, :, :])
        f_I_alpha=(f_I_alpha_0+f_I_alpha_1+f_I_alpha_2)
        f_I_alpha=self.pre1(f_I_alpha)
        f_I = self.pre2(I)
        f_delta_I = self.pre3(delta_I)
        res = self.pdu(f_I_alpha, f_I, f_delta_I)
        #res = self.post_precess(res)
        return f_I_alpha_0,f_I_alpha_1,f_I_alpha_2,f_I_alpha,f_I,f_delta_I,res

class Group2(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group2, self).__init__()
        
        pre_precess1_0 = [
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True)]
        self.pre1_0 = nn.Sequential(*pre_precess1_0)
        pre_precess1_1 = [
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True)]
        self.pre1_1 = nn.Sequential(*pre_precess1_1)
        pre_precess1_2 = [
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True)]
        self.pre1_2 = nn.Sequential(*pre_precess1_2)
        pre_precess1 = [
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True)]
        self.pre1 = nn.Sequential(*pre_precess1)

        pre_precess2 = [
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size), nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size), nn.ReLU(inplace=True)]
        self.pre2 = nn.Sequential(*pre_precess2)
        pre_precess3 = [
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size),nn.ReLU(inplace=True),
            conv(dim, dim, kernel_size), nn.ReLU(inplace=True)]
        self.pre3 = nn.Sequential(*pre_precess3)

        self.pdu = PDU(dim)

    def forward(self, I_alpha_0, I_alpha_1,I_alpha_2,I, delta_I):

        f_I_alpha_0 = self.pre1_0(I_alpha_0)
        f_I_alpha_1 = self.pre1_0(I_alpha_1)
        f_I_alpha_2 = self.pre1_0(I_alpha_2)
        f_I_alpha=(f_I_alpha_0+f_I_alpha_1+f_I_alpha_2)
        f_I_alpha= self.pre1(f_I_alpha)
        f_I = self.pre2(I)
        f_delta_I = self.pre3(delta_I)
        res = self.pdu(f_I_alpha, f_I, f_delta_I)
        return f_I_alpha_0,f_I_alpha_1,f_I_alpha_2,f_I,f_delta_I,res

class FusionModule(nn.Module):
    def __init__(self, n_feat, kernel_size=5):
        super(FusionModule, self).__init__()
        print("Creating BRB-Fusion-Module")
        self.block1 = blocks.BinResBlock(n_feat, kernel_size=kernel_size)
        self.block2 = blocks.BinResBlock(n_feat, kernel_size=kernel_size)

    def forward(self, x, y,z):
        #H_0 = x + y
        H_0 = x + y + z
        x_1, y_1, z_1, H_1 = self.block1(x, y, z, H_0)
        x_2, y_2, z_2, H_2 = self.block2(x_1, y_1, z_1, H_1)
        #x_1, y_1,H_1 = self.block1(x, y, H_0)
        #x_2, y_2,H_2 = self.block2(x_1, y_1, H_1)

        return H_2

class SemanticFusionUnit(nn.Module):
    def __init__(self, channels):
        super(SemanticFusionUnit, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels+3, out_channels=channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()#nn.Sigmoid()
        )

        #TODO: WEIGHT initial 【x】
        #TODO: 模块嵌入【预处理的数据】->dataloader 【x】
        #TODO：TEST的流程【SAM】
        #TODO：数据归一化，因为涉及加法 ->dataloader【x】
        
    def forward(self, fea, sem):
        #sem=self.conv0(sem)

        cat = torch.cat((fea, sem), dim = 1) # (b, c, h, w)
        fusion = self.conv(cat)

        return fusion
        

from fuse_block import TransformerBlock
class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        channels_2=3
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.fusion = SemanticFusionUnit(channels)
        #self.fusion = TransformerBlock(channels_2, channels, num_heads=64)#3

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()#nn.Sigmoid()
        )

    def forward(self, input, sem):
        # fea = self.in_conv(input)
        # fea = fea + self.fusion(fea, sem)
        # for conv in self.blocks:
        #     fea = fea + conv(fea)
        # fea = self.out_conv(fea)
        # illu = fea + input
        # illu = torch.clamp(illu, 0.0001, 1)

        fea=self.fusion(input, sem)

        return fea


class C2PNet(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(C2PNet, self).__init__()
        self.gps = gps
        self.channels = 64
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        assert self.gps == 3
        self.g1 = Group1(conv, self.channels, 3, blocks=blocks)#kernel_size
        self.g2 = Group2(conv, self.channels, 5, blocks=blocks)#kernel_size
        self.g3 = Group2(conv, self.channels, 5, blocks=blocks)  # kernel_size

        # self.ca = nn.Sequential(*[
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(self.dim * self.gps, self.dim // 4, 1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.dim // 4, self.dim * self.gps, 1, padding=0, bias=True),
        #     nn.Sigmoid()
        # ])
        post_precess = [
            conv(self.channels,self.channels, kernel_size),nn.ReLU(inplace=True),conv(self.channels,3, kernel_size),nn.ReLU(inplace=True)]
        self.post = nn.Sequential(*post_precess)
        
        self.fusion_feat = FusionModule(n_feat=64, kernel_size=5)

        self.enhance = EnhanceNetwork(layers=3, channels=64)

        self.sf_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels+3, out_channels=self.channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.sf_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels+3, out_channels=self.channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.sf_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels+3, out_channels=self.channels, kernel_size=5, stride=1,  padding=2),
            nn.ReLU(inplace=True),
        )
        self.sem_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.sem_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.sem_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1,  padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, I_alpha, I, delta_I, sem):

        f_I_alpha_0,f_I_alpha_1,f_I_alpha_2,res1_I_alpha, res1_I, res1_delta_I,res1_J = self.g1( I_alpha, I, delta_I)
        res2_I_alpha_0,res2_I_alpha_1,res2_I_alpha_2, res2_I, res2_delta_I,res2_J= self.g2(f_I_alpha_0,f_I_alpha_1,f_I_alpha_2, res1_I, res1_delta_I)
        res3_I_alpha_0,res3_I_alpha_1,res3_I_alpha_2, res3_I, res3_delta_I,res3_J=self.g3(res2_I_alpha_0,res2_I_alpha_1,res2_I_alpha_2,res2_I,res2_delta_I)

        sm0 = self.sem_conv0( sem / 255.)
        sm1 = self.sem_conv1( sem / 255.)
        sm2 = self.sem_conv2( sem / 255.)

        sf0 = self.sf_conv0(torch.cat((res1_J,sm0),1))
        sf1 = self.sf_conv1(torch.cat((res2_J,sm1),1))
        sf2 = self.sf_conv2(torch.cat((res3_J,sm2), 1))

        fusioned_feat = self.fusion_feat(sf0, sf1,sf2)
        out = self.post(fusioned_feat)


        #消融实验1
        #fusioned_feat = self.fusion_feat(res1_J, res2_J,res3_J)
        #out = self.post(fusioned_feat)


        return out


if __name__ == "__main__":
    net = C2PNet(gps=3, blocks=19)
    print(net)
