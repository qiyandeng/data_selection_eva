# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import numpy as np

# class DSN(nn.Module):
# 	"""Deep Summarization Network"""
# 	def __init__(self,in_dim =19,hid_dim=256, num_layers=1):
# 		super(DSN, self).__init__()
# 		# in_dim = in_dim*in_dim # for semantic segementation
# 		in_dim = in_dim 
# 		self.lstm = nn.LSTM(in_dim, hid_dim, num_layers=1, bidirectional=True, batch_first=True)
# 		self.fc2 = nn.Linear(hid_dim*2,hid_dim*2)
# 		self.fc1 = nn.Linear(hid_dim*2,1)
		

# 	def forward(self, x,in_dim=19):
		
# 		# x=torch.reshape(x,(x.shape[0],x.shape[1],in_dim*in_dim)) # for semantic segmentation
# 		x = x.float()
# 		x,_=self.lstm(x)
# 		x= self.fc2(x)
# 		x= self.fc1(x)
# 		p = F.sigmoid(x)
# 		return p


import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, hid_dim=256, num_layers=1):
        super(DSN, self).__init__()
        self.lstm = None  # LSTM 初始化为空
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.fc2 = nn.Linear(hid_dim * 2, hid_dim * 2)
        self.fc1 = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        # 动态初始化 LSTM
        if self.lstm is None:
            in_dim = x.size(-1)  # 获取输入数据的最后一个维度
            self.lstm = nn.LSTM(in_dim, self.hid_dim, num_layers=self.num_layers, 
                                bidirectional=True, batch_first=True).to(x.device)

        x = x.float()
        x, _ = self.lstm(x)
        x = self.fc2(x)
        x = self.fc1(x)
        p = F.sigmoid(x)
        return p