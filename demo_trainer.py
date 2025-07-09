# Huggingface Trainer DEMO --- with MINST dataset:

# To start DDP (Distributed Data Parallel) training with Huggingface Trainer, 
# you typically launch your script using torchrun or python -m torch.distributed.run.
# Example command to run this script with 2 GPUs:
# torchrun --nproc_per_node=2 demo_trainer.py

# 1. get the dataset ready:
import torch
from torchvision import datasets, transforms


data_directory = './dataset'  # this is the path to MINST dataset, already downloaded.


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# set download to be false so as to load local datasets
train_dset = datasets.MNIST(root=data_directory, train=True, download=False, transform=transform)
test_dset = datasets.MNIST(root=data_directory, train=False, download=False, transform=transform)


# 2. set the model:

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = F.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# 3. use the trainer:
from transformers import Trainer, TrainingArguments

model = BasicNet()

training_args = TrainingArguments(
    "basic-trainer",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    remove_unused_columns=False
)

def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"x":pixel_values, "labels":labels}

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["x"])
        target = inputs["labels"]
        loss = F.nll_loss(outputs, target)
        return (loss, outputs) if return_outputs else loss

trainer = MyTrainer(
    model,
    training_args,
    train_dataset=train_dset,
    eval_dataset=test_dset,
    data_collator=collate_fn,
)


# 4. run the training missions
trainer.train()



