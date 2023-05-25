import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def voxel_grid_sampling_1(point_cloud, sampling_factor):
    # Determine the minimum and maximum values for each axis
    B, N, C = point_cloud.shape
    device = point_cloud.device
    min_values = torch.min(point_cloud, dim=1)[0]
    max_values = torch.max(point_cloud, dim=1)[0]

    # Calculate the size of each voxel in each axis
    voxel_size = (max_values - min_values) / sampling_factor

    # Calculate the indices of the voxels that each point belongs to
    voxel_indices = ((point_cloud - min_values[:, None]) // voxel_size[:, None]).long()
    
    sampled_points = torch.zeros(B, N, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    a = torch.unique(voxel_indices, dim=1)
    sampled_mask = torch.ones(B, N, dtype=torch.long).to(device)
    indices = torch.nonzero(sampled_mask, as_tuple=False)
    output_tensor = torch.full_like(sampled_mask, -1)
    output_tensor[indices[:, 0], indices[:, 1]] = indices[:, 1]

    return output_tensor, sampled_mask

def dFPS(xyz, npoint, targets):
    device = xyz.device
    target_mask = (targets == 12).to(torch.long).to(device)
    f_pts = target_mask * 1
    b_pts = torch.logical_not(f_pts).to(torch.long).to(device)
    sampling_factor = 2.5
    output_tensor, sampled_mask = voxel_grid_sampling_1(xyz, 2.5)
    sampled_b_pts = sampled_mask & b_pts
    mask = torch.logical_or(f_pts.to(device), sampled_b_pts.to(device)).to(torch.long)
    
    B, N, C = xyz.shape
    sampled_indices = output_tensor
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    distance[mask == 0] = -1
    nni_mask = sampled_indices >= 0  
    non_negative_values = sampled_indices * nni_mask
    farthest = torch.tensor([row[torch.randint(0, len(row), (1,))].item() for row in non_negative_values], dtype = torch.long)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)  
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1] 
    
    sampled_targets = torch.zeros(B, npoint, dtype=torch.int64).to(device)

    for i in range(B):
        sampled_indices = centroids[i].to(device)
        sampled_targets[i] = targets[i, sampled_indices]

    return centroids , sampled_targets

def sample_and_group(npoint, radius, nsample, xyz, points, targets, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    device = xyz.device
    
    fps_idx, new_targets = dFPS(xyz, npoint, targets) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        fps_points = index_points(points, fps_idx)
        fps_points = torch.cat([new_xyz, fps_points], dim=-1)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
        fps_points = new_xyz
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_points, new_targets
    else:
        return new_xyz, new_points, new_targets


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class GraphAttention(nn.Module):
    def __init__(self,all_channel,feature_dim,dropout,alpha):
        super(GraphAttention, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(all_channel, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, center_xyz, center_feature, grouped_xyz, grouped_feature):
        '''
        Input:
            center_xyz: sampled points position data [B, npoint, C]
            center_feature: centered point feature [B, npoint, D]
            grouped_xyz: group xyz data [B, npoint, nsample, C]
            grouped_feature: sampled points feature [B, npoint, nsample, D]
        Return:
            graph_pooling: results of graph pooling [B, npoint, D]
        '''
        B, npoint, C = center_xyz.size()
        _, _, nsample, D = grouped_feature.size()
        delta_p = center_xyz.view(B, npoint, 1, C).expand(B, npoint, nsample, C) - grouped_xyz # [B, npoint, nsample, C]
        delta_h = center_feature.view(B, npoint, 1, D).expand(B, npoint, nsample, D) - grouped_feature # [B, npoint, nsample, D]
        delta_p_concat_h = torch.cat([delta_p,delta_h],dim = -1) # [B, npoint, nsample, C+D]
        e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a)) # [B, npoint, nsample,D]
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        attention = F.dropout(attention, self.dropout, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, grouped_feature),dim = 2) # [B, npoint, D]
        return graph_pooling


class GraphAttentionConvLayer(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all,droupout=0.6,alpha=0.2):
        super(GraphAttentionConvLayer, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.droupout = droupout
        self.alpha = alpha
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.GAT = GraphAttention(3+last_channel,last_channel,self.droupout,self.alpha)

    def forward(self, xyz, points, targets):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points, grouped_xyz, fps_points, new_targets = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, targets, True)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        # fps_points: [B, npoint, C+D,1]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        fps_points = fps_points.unsqueeze(3).permute(0, 2, 3, 1) # [B, C+D, 1,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            fps_points = F.relu(bn(conv(fps_points)))
            new_points =  F.relu(bn(conv(new_points)))
        # new_points: [B, F, nsample,npoint]
        # fps_points: [B, F, 1,npoint]
        new_points = self.GAT(center_xyz=new_xyz,
                              center_feature=fps_points.squeeze().permute(0,2,1),
                              grouped_xyz=grouped_xyz,
                              grouped_feature=new_points.permute(0,3,2,1))
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points, new_targets

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        return new_points

class NewPointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(NewPointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, points):
        points = points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points =  F.relu(bn(conv(points)))
        return points
    
class GACNet(nn.Module):
    def __init__(self, num_classes,droupout=0,alpha=0.2):
        super(GACNet, self).__init__()
        # GraphAttentionConvLayer: npoint, radius, nsample, in_channel, mlp, group_all,droupout,alpha
        self.sa1 = GraphAttentionConvLayer(1024, 0.1, 32, 6 + 3, [32, 32, 64], False, droupout,alpha)
        self.sa2 = GraphAttentionConvLayer(256, 0.2, 32, 64 + 3, [64, 64, 128], False, droupout,alpha)
        self.sa3 = GraphAttentionConvLayer(64, 0.4, 32, 128 + 3, [128, 128, 256], False, droupout,alpha)
        self.sa4 = GraphAttentionConvLayer(16, 0.8, 32, 256 + 3, [256, 256, 512], False, droupout,alpha)
        # PointNetFeaturePropagation: in_channel, mlp
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.nfp4 = NewPointNetFeaturePropagation(4096, [32, 32])
        self.dup_nfp4 = NewPointNetFeaturePropagation(128, [32, 32])
        self.nfp3 = NewPointNetFeaturePropagation(128, [64, 64])
        self.dup_nfp3 = NewPointNetFeaturePropagation(32, [64, 64])
        self.nfp2 = NewPointNetFeaturePropagation(64, [128, 128])
        self.dup_nfp2 = NewPointNetFeaturePropagation(64, [128, 128])
        self.nfp1 = NewPointNetFeaturePropagation(128, [300, 300, 300, 300])
        
        self.nfp0 = NewPointNetFeaturePropagation(4096, [32, 64, 128, 300])

        self.lin1 = nn.Linear(256, 300)
        self.lin2 = nn.Linear(300, 4096)

        self.conv1 = nn.Conv1d(300, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(droupout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def process(self, point):
        new_l1_points = self.nfp4(point)
        new_dup_l1_points = self.dup_nfp4(new_l1_points)
        new_dup_l1_points = nn.ConstantPad1d((0, 128-32), 0)(new_dup_l1_points)
        result1 = torch.add(new_l1_points, new_dup_l1_points)
        
        new_l2_points = self.nfp3(result1)
        new_dup_l2_points = self.dup_nfp3(new_l2_points)
        new_l2_points = nn.ConstantPad1d((0, 64-32), 0)(new_l2_points)
        result2 = torch.add(new_l2_points, new_dup_l2_points)
        
        new_l3_points = self.nfp2(result2)
        new_dup_l3_points = self.dup_nfp2(new_l3_points)
        new_l3_points = nn.ConstantPad1d((0, 128-64), 0)(new_l3_points)
        result3 = torch.add(new_l3_points, new_dup_l3_points)
        
        out1 = self.nfp1(result3)
        out2 = self.nfp0(point)
        
        out = torch.cat((out1, out2), 2)
        out = self.lin1(out)
        out = self.lin2(out)
        
        return out

    def forward(self, xyz, point, target):
        l1_xyz, l1_points, new_targets = self.sa1(xyz, point, target)
        l2_xyz, l2_points, new_targets = self.sa2(l1_xyz, l1_points, new_targets)
        l3_xyz, l3_points, new_targets = self.sa3(l2_xyz, l2_points, new_targets)
        l4_xyz, l4_points, new_targets  = self.sa4(l3_xyz, l3_points, new_targets)
        
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)
        all_points = self.process(l0_points)
        
        x = self.drop1(F.relu(self.bn1(self.conv1(all_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    model = GACNet(50)
    output = model(input)