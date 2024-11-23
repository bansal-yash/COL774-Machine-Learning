import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import pickle
import csv
from torchvision.transforms import ToPILImage
import sys

model_file = sys.argv[1]
test_file = sys.argv[2]
alpha_val = sys.argv[3]
gamma_val = sys.argv[4]

# python3 predict.py model.pth test.pkl alpha gamma


class StochasticDepthBasicBlock(torch.jit.ScriptModule):
    expansion = 1

    def __init__(self, p, in_channels, out_channels, stride=1):
        super().__init__()
        self.p = p

        # Added PReLU instead of ReLU
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        # Added SE block
        self.se = SEBlock(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

        # Added dropout for regularization
        self.dropout = nn.Dropout2d(0.1)

    def survival(self):
        return torch.bernoulli(torch.tensor(self.p).float())

    @torch.jit.script_method
    def forward(self, x):
        identity = self.shortcut(x)

        if self.training:
            if self.survival():
                out = self.residual(x)
                out = self.se(out)
                out = self.dropout(out)
                out = out + identity
            else:
                out = identity
        else:
            out = self.residual(x) * self.p
            out = self.se(out)
            out = out + identity

        return F.relu(out)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = self.log_softmax(preds)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction="mean")
        return (1 - self.epsilon) * nll + self.epsilon * (loss / n_classes)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal = FocalLoss(gamma=gamma)
        self.label_smoothing = LabelSmoothingCrossEntropy()

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        smooth_loss = self.label_smoothing(inputs, targets)
        return self.alpha * focal_loss + self.beta * smooth_loss


class ConfidenceAwareComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2, confidence_penalty=0.1):
        super().__init__()
        self.combo_loss = ComboLoss(alpha, beta, gamma)
        self.confidence_penalty = confidence_penalty

    def forward(self, logits, targets):
        combo_loss = self.combo_loss(logits, targets)
        probs = F.softmax(logits, dim=1)
        confidence = probs.gather(1, targets.unsqueeze(1)).squeeze()
        confidence_penalty = -self.confidence_penalty * torch.log(confidence)
        return combo_loss + confidence_penalty.mean()


def train_model(model, train_loader, epochs=300, device="cuda"):

    criterion = ConfidenceAwareComboLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        # Progress bar for training
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs} (Training)",
            unit="batch",
        ) as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix(
                    {
                        "Loss": f"{train_loss / (pbar.n + 1):.3f}",
                        "Acc": f"{100. * correct / total:.2f}%",
                    }
                )
                pbar.update(1)

        if epoch > 200:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

        scheduler.step()

        train_acc = 100.0 * correct / total

        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.3f}, Train Acc: {train_acc:.3f}%"
        )


class StochasticDepthResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, p_l=0.5):
        super(StochasticDepthResNet, self).__init__()
        self.in_channels = 64

        # Calculate linear decay survival probabilities
        self.num_blocks = sum(num_blocks)
        self.layer_positions = []
        current_position = 0
        for stage_blocks in num_blocks:
            for _ in range(stage_blocks):
                self.layer_positions.append(current_position)
                current_position += 1

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.PReLU()  # Changed to PReLU

        # ResNet stages
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, p_l=p_l)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, p_l=p_l)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, p_l=p_l)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, p_l=p_l)

        # Final classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, p_l):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i, stride in enumerate(strides):
            # Calculate survival probability for this layer
            layer_position = self.layer_positions[len(layers)]
            p_i = 1 - (layer_position / self.num_blocks) * (1 - p_l)

            layers.append(block(p_i, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        out = self.relu(self.bn1(self.conv1(x)))

        # ResNet stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Classification
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def stochastic_depth_resnet18(num_classes=100, p_l=0.5):
    """
    Create a ResNet-18 model with stochastic depth
    Args:
        num_classes: number of output classes
        p_l: target survival probability for the last layer
    """
    return StochasticDepthResNet(
        StochasticDepthBasicBlock, [2, 2, 2, 2], num_classes=num_classes, p_l=p_l
    )


def stochastic_depth_resnet34(num_classes=100, p_l=0.5):
    """
    Create a ResNet-34 model with stochastic depth
    Args:
        num_classes: number of output classes
        p_l: target survival probability for the last layer
    """
    return StochasticDepthResNet(
        StochasticDepthBasicBlock, [3, 4, 6, 3], num_classes=num_classes, p_l=p_l
    )


class CIFAR100Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, "rb") as f:
            self.data = pickle.load(f)
            self.transform = transform
            self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform != None:
            image = self.to_pil(image)
            image = self.transform(image)
        return image, label


torch.manual_seed(42)
torch.backends.cudnn.deterministic = False  # Set to False for better performance
torch.backends.cudnn.benchmark = True  # Enable benchmarking

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        ),
    ]
)

batch_size = 128

test_dataset = CIFAR100Dataset(test_file, test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = stochastic_depth_resnet18()
model.to(device)
model.load_state_dict(torch.load("model.pth"))


predictions = []
test_id_array = []

model.eval()
not_pred_count = 0
test_loss, test_correct, test_total = 0, 0, 0

with torch.no_grad():
    for images, test_ids in test_loader:
        images = images.to(device)

        outputs = model(images)
        outputs = outputs.squeeze()
        probs = torch.softmax(outputs, dim=1)
        max_probs, preds = torch.max(probs, 1)
        preds = preds.cpu().numpy()
        max_probs = max_probs.cpu().numpy()

        for i in range(len(preds)):
            if max_probs[i] < 0.57:
                preds[i] = -1
                not_pred_count += 1

        predictions.extend(preds)
        test_id_array.extend(test_ids)

print(not_pred_count)

answer_array = [["ID", "Predicted_label"]]

for i in range(len(test_id_array)):
    answer_array.append([test_id_array[i].item(), predictions[i]])

with open("submission.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(answer_array)

print("predictions done")
