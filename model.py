import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self, image_input_dim, text_input_dim, label_embedding_dim, num_labels, label_embedding_matrix):
        super(FeatureExtractor, self).__init__()

        self.image_cnn = nn.Sequential(
            nn.Conv2d(image_input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 1024)
        )

        self.text_mlp = nn.Sequential(
            nn.Linear(text_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )

        self.label_embedding_matrix = label_embedding_matrix
        self.num_labels = num_labels
        self.label_embedding_dim = label_embedding_dim

        self.label_weight_matrix = nn.Parameter(torch.randn(num_labels, label_embedding_dim))

    def forward(self, image, text, multi_label_matrix):
        image_features = self.image_cnn(image)
        image_features = image_features.view(image_features.size(0), -1)

        text_features = self.text_mlp(text)

        label_embeddings = torch.matmul(self.label_embedding_matrix, self.label_weight_matrix.t())
        multi_label_matrix = multi_label_matrix.float()

        multi_label_features = torch.matmul(multi_label_matrix, label_embeddings)

        return image_features, text_features, multi_label_features




feature_extractor = FeatureExtractor(image_input_dim=3, text_input_dim=300, label_embedding_dim=100, num_labels=24,
                                     label_embedding_matrix=label_embedding_matrix)

image_features, text_features, multi_label_features = feature_extractor(image_input, text_input, multi_label_matrix)


class ICCA(nn.Module):
    def __init__(self, image_input_dim=1024, text_input_dim=1024, multi_label_dim_features=1024, num_labels=24,
                 alpha=0.5, hidden_dim=64, num_layers=2, minus_one_dim=1024, in_channel=300):
        super(ICCA, self).__init__()
        self.alpha = alpha
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.gc1 = GCNLayer(in_channel, minus_one_dim)
        self.gc2 = GCNLayer(minus_one_dim, minus_one_dim)
        self.gc3 = GCNLayer(minus_one_dim, minus_one_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.hypo = nn.Linear(3 * minus_one_dim, minus_one_dim)

        self.feature_extractor = FeatureExtractor(image_input_dim, text_input_dim, multi_label_dim_features, num_labels,
                                                  label_embedding_matrix)

        self.gcn_layers = nn.ModuleList([GCNLayer(multi_label_dim_features + 1024, hidden_dim)] +
                                        [GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])

    def build_adjacency_matrix(self, multi_label_features, image_features, text_features):
        N = multi_label_features.shape[0]
        adj_matrix = torch.zeros((N, N))

        cosine_sim_matrix_image = torch.cosine_similarity(image_features.unsqueeze(0), image_features.unsqueeze(1), dim=-1)
        cosine_sim_matrix_text = torch.cosine_similarity(text_features.unsqueeze(0), text_features.unsqueeze(1), dim=-1)

        for i in range(N):
            for j in range(N):
                shared_labels = torch.sum(multi_label_features[i] * multi_label_features[j])

                adj_matrix[i, j] = (cosine_sim_matrix_image[i, j] + cosine_sim_matrix_text[i, j] +
                                    (shared_labels / multi_label_features.shape[1]) * self.alpha)

        return adj_matrix

    def forward(self, image, text, multi_label_matrix):
        image_features, text_features, multi_label_features = self.feature_extractor(image, text, multi_label_matrix)
        adj_matrix = self.build_adjacency_matrix(multi_label_features, image_features, text_features)

        feature_matrix = torch.cat([image_features, text_features, multi_label_features], dim=1)

        x = feature_matrix
        for layer in self.gcn_layers:
            x = layer(adj_matrix, x)

        return x


model = ICCA(image_input_dim=1024, text_input_dim=1024, multi_label_dim_features=1024, num_labels=24, alpha=0.5)

semantic_enhanced_representation = model(image_input, text_input, multi_label_matrix)


class SharedClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SharedClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class CrossModalAttention(nn.Module):
    def __init__(self, d, num_labels):
        super(CrossModalAttention, self).__init__()
        self.fc_v = nn.Linear(d, d)
        self.fc_t = nn.Linear(d, d)
        self.fc_m = nn.Linear(num_labels, d)

    def forward(self, ziv, zti, multi_label_embeddings):
        miv = torch.sigmoid(self.fc_v(ziv))
        mti = torch.sigmoid(self.fc_t(zti))

        mimv = torch.sigmoid(self.fc_m(multi_label_embeddings))
        mimt = torch.sigmoid(self.fc_m(multi_label_embeddings))

        z_hat_ti = miv * zti
        z_hat_iv = mti * ziv
        zimv = mimv * z_hat_iv
        zimt = mimt * z_hat_ti

        z_hat_image = torch.cat([zimv], dim=-1)
        z_hat_text = torch.cat([zimt], dim=-1)

        return z_hat_image, z_hat_text


class StudentModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentModule, self).__init__()

        self.fc_image = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.fc_text = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, image_input, text_input):
        image_features = self.fc_image(image_input)
        text_features = self.fc_text(text_input)
        return image_features, text_features