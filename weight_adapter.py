import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json

class FeatureDataset(Dataset):
    def __init__(self, data_json, num_object, label_offset=0):
        if os.path.exists(data_json):
            if data_json.endswith('.json'):
                with open(data_json, 'r') as f:
                    feat_dict = json.load(f)
                self.data = torch.Tensor(feat_dict['features'])#.cuda()
            elif data_json.endswith('.pth'):
                features = torch.load(data_json)
                emb_dim = features.size(-1)
                self.data = features.view(-1, emb_dim).float().cuda()
                print("Shape of descriptor tensor: ", self.data.size())
        self.num_template_per_object = self.data.size(0) // num_object
        print(f'num_template_per_object: {self.num_template_per_object}')
        self.label_offset = label_offset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_feature = self.data[index]
        label = index // self.num_template_per_object + self.label_offset  # 100 objects in total

        return img_feature, label

class WeightAdapter(nn.Module):
    """
    Predict weights for each feature vector.
    """
    def __init__(self, c_in, reduction=4, scalar=10.0):
        """

        @param c_in: The channel size of the input feature vector
        @param reduction: the reduction factor for the hidden layer
        @param scalar: A scalar to scale the input feature vector
        """
        super(WeightAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        # self.ratio = ratio
        self.scalar = scalar

    def forward(self, inputs):
        # inputs = F.normalize(inputs, dim=-1, p=2)
        inputs = self.scalar * inputs
        x = self.fc(inputs)
        x = x.sigmoid()
        x = x * inputs

        return x


# modified from SimCLR loss: https://github.com/sthalles/SimCLR/blob/master/simclr.py#L26
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5, eps=1e-8):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.eps = eps  # Small constant to avoid division by zero or log(0)

    def forward(self, features, labels):
        # original_labels = labels
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float()
        mask_negative = 1 - mask_positive
        mask_positive.fill_diagonal_(0)

        exp_logits = torch.exp(similarity_matrix / self.temperature)
        sum_negatives = torch.sum(exp_logits * mask_negative, dim=1, keepdim=True) + self.eps # replace 1-mask_positive with mask_negative
        sum_positives = torch.sum(exp_logits * mask_positive, dim=1, keepdim=True)

        # Adding eps inside the log to avoid log(0)
        loss = -torch.log(sum_positives / (sum_positives + sum_negatives) + self.eps)
        loss = loss.mean()

        return loss


if __name__ == '__main__':
    adapter_type = 'weight'
    dataset_name = f'insDet_{adapter_type}_0523'
    temperature = 0.05
    ratio = 0.6
    feature_dataset = FeatureDataset(data_json='./object_features.json', num_object=1)
    # Assuming 'features' is your (N, 1024) tensor
    batch_size = 1024

    cur_feature_dataset = feature_dataset

    # Example training loop
    input_features = 1024  # Size of the input feature vector, 1024 for large, 768 for base, 384 for small
    reduction = 4 # Reduction factor for the hidden layer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 1e-3
    model = WeightAdapter(input_features, reduction=reduction).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) #
    criterion = InfoNCELoss(temperature=temperature).to(device)
    epochs = 40

    dataloader = DataLoader(cur_feature_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):

        for inputs, labels in dataloader: # in dataloader: tqdm(dataloader)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    save_model = True
    adapter_args = f'{dataset_name}_temp_{temperature}_epoch_{epochs}_lr_{learning_rate}_bs_{batch_size}_vec_reduction_{reduction}'
    os.makedirs('adapter_weights', exist_ok=True)
    if save_model:
        model_path = f'adapter_weights/{adapter_args}_weights.pth'
        # Save the model state dictionary
        torch.save(model.state_dict(), model_path)

        print(f'Model weights saved to {model_path}')

    save_features = True
    if save_features:
        # Assuming model is already defined and loaded with trained weights
        model.eval()  # Set the model to evaluation mode
        batch_size = 64
        test_dataloader = DataLoader(cur_feature_dataset, batch_size=batch_size, shuffle=False)

        adatped_features = []
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            # labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                # Perform inference using the model
                # Your inference code here
                adatped_features.append(outputs)
        adatped_features = torch.cat(adatped_features, dim=0)
        print(adatped_features.size())
        feat_dict = dict()
        feat_dict['features'] = adatped_features.detach().cpu().tolist()
        output_dir = './adapted_obj_feats'
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        json_filename = f'{adapter_args}.json'
        with open(os.path.join(output_dir, json_filename), 'w') as f:
            json.dump(feat_dict, f)
        print(f"saving adapted features {os.path.join(output_dir, json_filename)}")


