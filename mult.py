import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, datasets
from download_data import download_data
from os import cpu_count
from pathlib import Path
from multiprocessing import freeze_support

SEED = 40
BATCH_SIZE = 8
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

data_10_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                                     destination="pizza_steak_sushi")

data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")

def get_train_dir(dir): return dir / "train"
def get_test_dir(dir):  return dir / "test"

def create_model(model_name, device):
    if model_name == "effnetb0":
        return create_effnetb0(device)
    if model_name == "effnetb2":
        return create_effnetb2(device)
    if model_name == "shufflenetv2x05":
        return create_shufflenetv2x05(device)

def create_effnetb0(device):
    effnetb0_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    effnetb0 = torchvision.models.efficientnet_b0(weights=effnetb0_weights).to(device)
    return effnetb0

def create_effnetb2(device):
    effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    effnetb2 = torchvision.models.efficientnet_b2(weights=effnetb2_weights).to(device)
    return effnetb2

def create_shufflenetv2x05(device):
    shufflenet_weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT
    shufflenet = torchvision.models.shufflenet_v2_x0_5(weights=shufflenet_weights).to(device)
    return shufflenet

effnetb0_transform = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms()
effnetb2_transform = torchvision.models.EfficientNet_B2_Weights.DEFAULT.transforms()
shufflenet_transform = torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms()

def get_dataloaders(dir, transform, batch_size):
    train_data = datasets.ImageFolder(
        root=get_train_dir(dir),
        transform=transform,
    )
    test_data = datasets.ImageFolder(
        root=get_test_dir(dir),
        transform=transform
    )

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=cpu_count(),
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=cpu_count(),
        shuffle=False
    )

    return train_dataloader, test_dataloader, train_data.classes

datasets_dirs = {
    "data_10_percent": data_10_percent_path,
    "data_20_percent": data_20_percent_path
}

model_names = [
    "effnetb0",
    "effnetb2",
    "shufflenetv2x05"
]

model_transform = {
    "effnetb0": effnetb0_transform,
    "effnetb2": effnetb2_transform,
    "shufflenetv2x05": shufflenet_transform
}

epochs_num = [5, 10]

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device, writer):
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        model.train()

        train_loss, train_acc = 0, 0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)

            loss = loss_fn(y_logits, y)
            train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            train_acc = (y_pred == y).sum().item()/len(y_pred)
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)

                test_y_logits = model(X)

                loss = loss_fn(test_y_logits, y)
                test_loss += loss.item()

                test_pred = torch.argmax(torch.softmax(test_y_logits, dim=1), dim=1)
                test_acc += (test_pred == y).sum().item()/len(test_pred)
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss,
                                "test_loss": test_loss   },
                global_step=epoch
            )

            writer.add_scalars(
                main_tag="Accuracy", 
                tag_scalar_dict={"train_acc": train_acc,
                                "test_acc": test_acc   }, 
                global_step=epoch
            )

            #writer.add_graph(model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device))

            writer.close()
    torch.cuda.empty_cache()
    return results



if __name__ == '__main__':
    freeze_support()
    for model_name in model_names:
        for (dataset_name, dir) in datasets_dirs.items():
            for epochs in epochs_num:
                experiment_name = f"07_{model_name}_{dataset_name}_{epochs}_epochs.pth"

                model = create_model(model_name, device)

                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

                train_dataloader, test_dataloader, classes = get_dataloaders(
                    transform=model_transform[model_name],
                    dir=dir,
                    batch_size=BATCH_SIZE
                )

                train(
                    model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=epochs,
                    device=device,
                    writer=SummaryWriter(log_dir=f"runs2\\07_{model_name}_{dataset_name}_{epochs}_epochs")
                )

                
                save_folder = Path("models")
                save_folder.mkdir(parents=True, exist_ok=True)
                save_path = save_folder / experiment_name
                torch.save(obj=model.state_dict(), f=save_path)
                print(torch.cuda.memory_summary(device=device, abbreviated=False))