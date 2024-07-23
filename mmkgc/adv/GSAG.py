import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attention_weights = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float()))
        out = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.out(out)
        return out

class GatedFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, proj_dim):
        super(GatedFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim1, proj_dim)
        self.fc2 = nn.Linear(input_dim2, proj_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2, proj_dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=-1)
        gate = self.gate(combined)
        return gate * self.fc1(x1) + (1 - gate) * self.fc2(x2)

class BaseGenerator(nn.Module):
    def __init__(self, noise_dim, structure_dim, img_dim):
        super(BaseGenerator, self).__init__()
        self.proj_dim = 512
        self.noise_dim = noise_dim
        self.structure_dim = structure_dim
        self.img_dim = img_dim
        self.attention1 = MultiHeadSelfAttention(self.proj_dim)
        self.attention2 = MultiHeadSelfAttention(self.proj_dim)
        self.fusion1 = GatedFusion(noise_dim, structure_dim, self.proj_dim)
        self.fusion2 = GatedFusion(self.proj_dim, self.proj_dim, self.proj_dim)
        self.generator_model = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim),
            self.attention1,
            nn.LeakyReLU(),
            self.attention2,
            nn.LeakyReLU(),
            nn.Linear(self.proj_dim, img_dim)
        )

    def forward(self, batch_ent_emb):
        random_noise = torch.randn((batch_ent_emb.shape[0], self.noise_dim)).cuda()
        fused1 = self.fusion1(random_noise, batch_ent_emb)
        fused2 = self.fusion2(fused1, fused1)
        out = self.generator_model(fused2)
        out = out.squeeze(1)
        return out

class RandomGenerator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super(RandomGenerator, self).__init__()
        self.proj_dim = 256
        self.noise_dim = noise_dim
        self.generator_model = nn.Sequential(
            nn.Linear(noise_dim, self.proj_dim),
            nn.LeakyReLU(),
            nn.Linear(self.proj_dim, img_dim)
        )

    def forward(self, batch_ent_emb):
        random_noise = torch.randn((batch_ent_emb.shape[0], self.noise_dim)).cuda()
        out = self.generator_model(random_noise)
        return out

class MultiGenerator(nn.Module):
    def __init__(self, noise_dim, structure_dim, img_dim):
        super(MultiGenerator, self).__init__()
        self.img_generator = BaseGenerator(noise_dim, structure_dim, img_dim)
        self.text_generator = BaseGenerator(noise_dim, structure_dim, img_dim)

    def forward(self, batch_ent_emb, modal):
        if modal == 1:
            return self.img_generator(batch_ent_emb)
        elif modal == 2:
            return self.text_generator(batch_ent_emb)
        else:
            raise NotImplementedError

class Similarity(nn.Module):
    def __init__(self, temp):
        super(Similarity, self).__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super(ContrastiveLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, node_emb, img_emb):
        batch_sim = self.sim_func(node_emb.unsqueeze(1), img_emb.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)
