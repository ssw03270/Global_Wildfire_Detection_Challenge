import os
import argparse
import torch
from torch.utils.data import DataLoader

import joblib
import numpy as np
import random
from tqdm import tqdm

from model import SwinTransformer
from dataloader import ImageDatasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize a transformer with user-defined hyperparameters.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max_epoch", type=int, default=1000, help="Maximum number of epochs for training.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used in the transformer model.")
    parser.add_argument("--seed", type=int, default=327, help="Random seed for reproducibility across runs.")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use tensorboard.")
    parser.add_argument("--use_checkpoint", type=bool, default=False, help="Use checkpoint model.")
    parser.add_argument("--checkpoint_epoch", type=int, default=0, help="Use checkpoint index.")
    parser.add_argument("--val_epoch", type=int, default=1, help="Use checkpoint index.")
    parser.add_argument("--save_epoch", type=int, default=10, help="Use checkpoint index.")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--save_dir_path", type=str, default="transformer_graph", help="save dir path")
    parser.add_argument("--lr", type=float, default=3e-5, help="save dir path")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="save dir path")

    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    device = torch.device('cpu')

    test_dataset = ImageDatasets(data_type='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    transformer = SwinTransformer(
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=10,
            img_size=256,
            head_dim=32,
            window_size=8,
            downscaling_factors=(2, 2, 2, 2),
            relative_pos_embedding=True
        ).to(device=device)

    checkpoint = torch.load("./models/" + opt.save_dir_path + "/epoch_"+ str(opt.checkpoint_epoch) + ".pth")
    transformer.load_state_dict(checkpoint['model_state_dict'])

    transformer.eval()
    y_predict = {}
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader)):
            input_img = data['input_img'].to(device=device)

            output = transformer(input_img)
            output = output.detach().cpu().numpy()
            output = np.where(output[0, :, :, 0] > 0.5, 1, 0)
            output = output.astype(np.uint8)

            key_name = data['key_name'][0]
            y_predict[key_name] = output
            print(np.sum(output))

    joblib.dump(y_predict, './y_pred.pkl')


