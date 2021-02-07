class DanQ(torch.nn.Module):
    def __init__(self):
        self.a = "1"

    def forward(self, x):
        pass

    def __str__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return "a string represeting this model"