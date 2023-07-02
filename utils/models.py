import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, backbone, head):
        super(CustomModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

    def freeze_backbone(self, mode="transfer-learning"):
        if mode == "transfer-learning":
            # With the custom model we can target just the backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif mode == "fine-tune":
            for name, param in self.named_parameters():
                param.requires_grad = True
                if "bn" in name or "batchnorm" in name.lower():
                    param.requires_grad = False
        else:
            print("Provide a valid mode: 'transfer-learning' or 'fine-tune'")
