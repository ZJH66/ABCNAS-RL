import os
import torch
import torch.nn as nn
import torch.distributions as distr
import torch.nn.functional as F



# # (2) Decoder
# class Decoder(nn.Module):
#     def __init__(self, seq_len, no_features, output_size, config):
#         super().__init__()
#         self.max_length = seq_len
#         self.config = config
#         self.seq_len = seq_len
#         self.no_features = no_features
#         self.hidden_size = (2 * no_features)
#         self.output_size = output_size
#         self.LSTM1 = nn.LSTM(
#             input_size=no_features,
#             hidden_size=no_features,
#             num_layers=3,
#             batch_first=True,
#             bidirectional=True
#         ).cuda(0)
#
#         self.fc = nn.Linear(self.hidden_size, output_size).cuda(0)
#
#     def forward(self, x):
#         # x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
#         x, (hidden_state, cell_state) = self.LSTM1(x)
#         x = x.reshape((-1, self.seq_len, self.hidden_size))
#         out = self.fc(x)
#         self.adj_prob = out
#
#         self.mask = 0
#         self.samples = []
#         self.mask_scores = []
#         self.entropy = []
#
#         for i in range(self.max_length):
#             position = torch.ones([x.shape[0]]) * i
#             position = position.type(torch.LongTensor)
#             # if self.config.device_type == 'gpu':
#             #     position = position.cuda(self.config.device_ids)
#             # Update mask
#             self.mask = torch.zeros(x.shape[0], self.max_length).scatter_(1, position.view(
#                 x.shape[0], 1), 1)
#             if self.config.device_type == 'gpu':
#                 self.mask = self.mask.cuda(self.config.device_ids)
#
#             masked_score = self.adj_prob[:, i, :] - 100000000. * self.mask
#             prob = distr.Bernoulli(logits=masked_score)  # probs input probability, logit input log_probability
#
#             sampled_arr = prob.sample()  # Batch_size, seqlenght for just one node
#             sampled_arr.requires_grad = True
#
#             self.samples.append(sampled_arr)
#             self.mask_scores.append(masked_score)
#             self.entropy.append(prob.entropy())
#
#         return self.samples, self.mask_scores, self.entropy


class BiGRU_Attention(nn.Module):
    print("zjh")
    def __init__(self, config):
        super(BiGRU_Attention, self).__init__()
        self.config = config
        self.input_embed = config.hidden_dim
        self.lstm = self.LSTM1 = nn.LSTM(
            input_size=self.config.max_length,
            hidden_size=self.config.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        ).cuda(0)
        self.out = nn.Linear(self.config.hidden_dim * 2, self.config.max_length).cuda(0)
        self.emb = nn.Parameter(
            torch.Tensor(*(1, self.config.hidden_dim, self.config.hidden_dim)).cuda(self.config.device_ids))
        self.bn_layer = nn.BatchNorm1d(self.config.hidden_dim).cuda(self.config.device_ids)
        self.w_o = nn.Parameter(
            torch.Tensor(self.config.hidden_dim * 2, self.config.hidden_dim * 2).cuda(self.config.device_ids))
        self.u_o = nn.Parameter(torch.Tensor(self.config.hidden_dim * 2, 1).cuda(self.config.device_ids))
        self.fc = nn.Linear(self.config.hidden_dim * 2, self.config.max_length).cuda(0)
        self.conv1 = nn.Conv1d(in_channels=self.input_embed, out_channels=self.config.max_length, kernel_size=1,
                               bias=True).cuda(self.config.device_ids)

        self.Q_layer = nn.Sequential(nn.Linear(in_features=self.config.hidden_dim, out_features=self.config.hidden_dim),
                                     nn.ReLU()).cuda(self.config.device_ids)
        self.K_layer = nn.Sequential(nn.Linear(in_features=self.config.hidden_dim, out_features=self.config.hidden_dim),
                                     nn.ReLU()).cuda(self.config.device_ids)
        self.V_layer = nn.Sequential(nn.Linear(in_features=self.config.hidden_dim, out_features=self.config.hidden_dim),
                                     nn.ReLU()).cuda(self.config.device_ids)

        self.relu = nn.ReLU()

        nn.init.uniform_(self.w_o, -0.1, 0.1)
        nn.init.uniform_(self.u_o, -0.1, 0.1)
        nn.init.xavier_uniform_(self.emb)

    def att(self, inputs, dropout_rate=0.1):
        input_dimension = inputs.shape[1]
        inputs = inputs.permute(0, 2, 1)

        Q = self.Q_layer(inputs)  # [batch_size, seq_length, n_hidden]
        K = self.K_layer(inputs)  # [batch_size, seq_length, n_hidden]
        V = self.V_layer(inputs)  # [batch_size, seq_length, n_hidden]

        # Split and concat
        Q_ = torch.cat(torch.split(Q, int(input_dimension), dim=2),
                       dim=0)  # [batch_size, seq_length, n_hidden/num_heads]
        K_ = torch.cat(torch.split(K, int(input_dimension), dim=2),
                       dim=0)  # [batch_size, seq_length, n_hidden/num_heads]
        V_ = torch.cat(torch.split(V, int(input_dimension), dim=2),
                       dim=0)  # [batch_size, seq_length, n_hidden/num_heads]

        # Multiplication
        outputs = torch.matmul(Q_, K_.permute([0, 2, 1]))  # num_heads*[batch_size, seq_length, seq_length]

        # Scale
        outputs = outputs / (K_.shape[-1] ** 0.5)

        # Activation
        outputs = F.softmax(outputs, dim=1)  # num_heads*[batch_size, seq_length, seq_length]

        # Dropouts
        outputs = F.dropout(outputs, p=dropout_rate, training=True)

        # Weighted sum
        outputs = torch.matmul(outputs, V_)  # num_heads*[batch_size, seq_length, n_hidden/num_heads]

        # Restore shape
        outputs = torch.cat(torch.split(outputs, int(outputs.shape[0]), dim=0),
                            dim=2)  # [batch_size, seq_length, n_hidden]

        # Residual connection
        outputs = outputs + inputs  # [batch_size, seq_length, n_hidden]

        outputs = outputs.permute(0, 2, 1)

        # Normalize
        outputs = self.bn_layer(outputs)  # [batch_size, seq_length, n_hidden]

        return outputs

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)

        # Embed input sequence
        W_embed = self.emb
        W_embed_ = W_embed.permute(2, 1, 0)
        self.embedded_input = F.conv1d(inputs, W_embed_, stride=1)

        # Batch Normalization
        self.enc = self.bn_layer(self.embedded_input)

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # output : [seq_len, batch_size, n_hidden * num_directions(=2)]
        # input = self.enc.transpose(1, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm(inputs)
        out = self.att(output)
        out = self.fc(out)
        out = self.relu(out)
        # u = torch.tanh_(torch.matmul(output, self.w_o))
        #
        # att = torch.matmul(u, self.u_o)
        # att_score = F.softmax(att, dim=1)
        # sx = output * att_score
        # out = torch.sum(sx, dim=1)
        # out = self.fc(sx)
        # out = self.relu(out)
        # out = out.unsqueeze(1).repeat(1, self.config.max_length, 1)
        self.adj_prob = out
        self.adj_prob.permute(0, 2, 1)

        self.mask = 0
        self.samples = []
        self.mask_scores = []
        self.entropy = []

        inputs = inputs.permute(0, 2, 1)
        for i in range(self.config.max_length):
            position = torch.ones([inputs.shape[0]]) * i
            position = position.type(torch.LongTensor)
            # if self.config.device_type == 'gpu':
            #     position = position.cuda(self.config.device_ids)
            # Update mask
            self.mask = torch.zeros(inputs.shape[0], self.config.max_length).scatter_(1, position.view(
                inputs.shape[0], 1), 1)
            if self.config.device_type == 'gpu':
                self.mask = self.mask.cuda(self.config.device_ids)

            masked_score = self.adj_prob[:, i, :] - 100000000. * self.mask
            prob = distr.Bernoulli(logits=masked_score)  # probs input probability, logit input log_probability

            sampled_arr = prob.sample()  # Batch_size, seqlenght for just one node
            sampled_arr.requires_grad = True

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy

