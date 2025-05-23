import numpy as np
import torch

from FGD.embedding_net import EmbeddingNet

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore warnings


class EmbeddingSpaceEvaluator:
    def __init__(self, embed_net_path, n_frames, device, pose_dim):
        # init embed net
        ckpt = torch.load(embed_net_path, map_location=device)
        self.pose_dim = pose_dim
        self.net = EmbeddingNet(self.pose_dim, n_frames).to(device)
        self.net.load_state_dict(ckpt)
        self.net.train(False)

        self.device = device
        self.reset()

    def reset(self):
        self.real_samples = []
        self.generate_samples = []
        self.real_feat_list = []
        self.generated_feat_list = []

    def get_no_of_samples(self):
        return len(self.real_feat_list)

    def push_real_samples(self, samples):
        feat, _ = self.net(samples)
        self.real_samples.append(samples.cpu().numpy().reshape(samples.shape[0], -1))
        self.real_feat_list.append(feat.data.cpu().numpy())

    def push_generated_samples(self, samples):
        feat, _ = self.net(samples)
        self.generate_samples.append(samples.cpu().numpy().reshape(samples.shape[0], -1))
        self.generated_feat_list.append(feat.data.cpu().numpy())

    def get_fgd(self, use_feat_space=True):
        if use_feat_space:
            generated_data = np.vstack(self.generated_feat_list)
            real_data = np.vstack(self.real_feat_list)
        else:
            generated_data = np.vstack(self.generate_samples)
            real_data = np.vstack(self.real_samples)

        frechet_dist = self.frechet_distance(generated_data, real_data)
        return frechet_dist

    def frechet_distance(self, samples_A, samples_B):
        print("Calculating means and covariances...")
        A_mu = np.mean(samples_A, axis=0)
        A_sigma = np.cov(samples_A, rowvar=False)
        B_mu = np.mean(samples_B, axis=0)
        B_sigma = np.cov(samples_B, rowvar=False)

        print("Calculating frechet distance with fast PyTorch sqrtm...")
        A_mu_torch = torch.from_numpy(A_mu).to(torch.float32).to(self.device)
        A_sigma_torch = torch.from_numpy(A_sigma).to(torch.float32).to(self.device)
        B_mu_torch = torch.from_numpy(B_mu).to(torch.float32).to(self.device)
        B_sigma_torch = torch.from_numpy(B_sigma).to(torch.float32).to(self.device)

        try:
            frechet_dist = self.torch_frechet_distance(A_mu_torch, A_sigma_torch,
                                                       B_mu_torch, B_sigma_torch)
        except Exception as e:
            print("Frechet distance failed:", e)
            frechet_dist = 1e+10
        return frechet_dist

    def torch_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        cov_prod = sigma1 @ sigma2

        covmean = self.matrix_sqrt_newton_schulz(cov_prod, eps=eps)

        if torch.is_complex(covmean):
            covmean = covmean.real  # in case of small imaginary parts

        tr_covmean = torch.trace(covmean)
        return (diff @ diff + torch.trace(sigma1) +
                torch.trace(sigma2) - 2 * tr_covmean).item()

    def matrix_sqrt_newton_schulz(self, A, num_iters=50, eps=1e-10):
        normA = A.norm()
        Y = A / normA
        I = torch.eye(A.size(0), device=A.device)
        Z = torch.eye(A.size(0), device=A.device)

        for i in range(num_iters):
            T = 0.5 * (3.0 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z
            if (Y @ Y - I).abs().max() < eps:
                break

        return Y * torch.sqrt(normA)