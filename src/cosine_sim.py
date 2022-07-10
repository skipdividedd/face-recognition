
import pandas as pd
import torch.nn as nn
import torch
device = 'cpu'


#resnet = train('resnet')


def pd_embeddings(model, train_loader, test_loader):

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    model.logits = Identity()

    def embed(loader, model):
        targets = []
        embeddings = []
        model.eval()
        with torch.no_grad():
            for data in loader:
                inputs, labels = data[0], data[1]

                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass
                output = model(inputs)

                embeddings.append(output.cpu())
                targets.append(labels.data.cpu())

        return torch.cat(embeddings).numpy(), torch.cat(targets).numpy()

    train_embeddings, train_targets = embed(train_loader, model)  # getting train embeddings and labels
    test_embeddings, test_targets = embed(test_loader, model)  # getting test embeddings and labels

    # Creating DataFrames to store data for train and test:
    train_dataset_emb = pd.DataFrame(
                            {'label': train_targets,
                            'images_train': list(train_embeddings)},
                            columns=['label', 'images_train']).sort_values('label').reset_index(drop=True)

    test_dataset_emb = pd.DataFrame(
                            {'label': test_targets,
                            'images_test': list(test_embeddings)},
                            columns=['label', 'images_test']).sort_values('label').reset_index(drop=True)

    train_dataset_emb['images_train'] = train_dataset_emb['images_train'].apply(lambda x: x.reshape(1, -1))
    test_dataset_emb['images_test'] = test_dataset_emb['images_test'].apply(lambda x: x.reshape(1, -1))

    return train_dataset_emb, test_dataset_emb



