import torch.nn as nn 
from torch_geometric.nn import Sequential, GATv2Conv


class ResGATv3(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=8, layer_num:int=2):
        super(ResGATv3, self).__init__()
        
        assert (layer_num>=0)
        self.layer_num = layer_num

        self._hidden_dim = hidden_dim
        self._input_dim = input_dim

        self.embed_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList([Sequential('x, edge_index', [
            (GATv2Conv(hidden_dim, hidden_dim, heads=heads), 'x, edge_index -> x'),
            nn.Linear(hidden_dim*heads, hidden_dim)
            ]) for i in range(layer_num)])
        
        self.fcs =  nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
            ) for i in range(layer_num)])

        self.leakyrelu = nn.LeakyReLU()

        self.gatnorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for i in range(layer_num)])
        self.fcnorms = nn.ModuleList([nn.LayerNorm(hidden_dim) for i in range(layer_num)])
        

    def forward(self, x, edge_index):
        x = self.embed_fc(x)
        
        for i in range(self.layer_num):
            residual = x
            x = self.convs[i](x, edge_index)
            x = self.gatnorms[i](x + residual)

            residual = x
            x = self.fcs[i](x)
            x = self.fcnorms[i](x + residual)

        y = x
        return y







class NetworkFinetuningModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=8):
        super(NetworkFinetuningModel, self).__init__()
        
        self.global_conv = ResGATv3(input_dim, hidden_dim, heads, layer_num=2)
        self.readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

    def forward(self, data):
        x = data.x 
        edge_index = data.edge_index
        
        # Graph attention do not need the sparse adj tensor
        # Could be used for for the GNN model which require sparse matrix multiply in message passing, i.e., sparse adj matrix
        # edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)).t()
        
        x = self.global_conv(x, edge_index)
        y = self.readout(x)
        return y