import argparse
import torch
from torch import nn
from tqdm import tqdm
from experiments import mnist_dataset
from difflogic import Conv, Logic, GroupSum
import os
from difflogic import CompiledLogicNet
from experiments.results_json import ResultsJSON


class MNISTArchitecture(nn.Module):
    def __init__(self, *,
                 model_scale: int = 16,
                 temperature: float = 6.5):
        super().__init__()

        self.output_gate_factor = 2 if model_scale <= 64 else 1
        self.temperature = temperature

        self.k = model_scale

        self.model = nn.Sequential(
            Conv(in_channels=1, out_channels=self.k, kernel_size=5, padding=0, stride=1, depth=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=self.k, out_channels=3 * self.k, kernel_size=2, padding=1, stride=1, depth=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(in_channels=3 * self.k, out_channels=9 * self.k, kernel_size=2, padding=1, stride=1, depth=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            Logic(in_dim=81 * self.k, out_dim=1280 * self.k * self.output_gate_factor),
            Logic(in_dim=1280 * self.k * self.output_gate_factor, out_dim=640 * self.k * self.output_gate_factor),
            Logic(in_dim=640 * self.k * self.output_gate_factor, out_dim=320 * self.k * self.output_gate_factor),
            GroupSum(k=10, tau=self.temperature)
        )

        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def prepare_images(self, images):
        images = torch.where(images != 0, torch.ones_like(images), torch.zeros_like(images))
        return images


def main(args):
    if args.experiment_id is not None:
        assert 520_000 <= args.experiment_id < 530_000, args.experiment_id
        results = ResultsJSON(eid=args.experiment_id, path='./results/')
        results.store_args(args)
        
    device = torch.device(args.device)
    pin_memory = (args.device == 'cuda')
    train_set = mnist_dataset.MNIST('./data-mnist', train=True, download=True, remove_border=False)
    test_set = mnist_dataset.MNIST('./data-mnist', train=False, remove_border=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory, drop_last=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, drop_last=True, num_workers=0)

    model = MNISTArchitecture(model_scale=args.model_scale, temperature=args.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = model.prepare_images(x).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = model.loss_function(y_hat, y)
            loss.backward()
            optimizer.step()

        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = model.prepare_images(x).to(device)
                y = y.to(device)
                y_hat = model(x)
                _, predicted = torch.max(y_hat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        print(f'Epoch {epoch}: Test Accuracy: {100 * correct / total}%')
        
    if args.compile_model:
        print('\n' + '='*80)
        print(' Converting the model to C code and compiling it...')
        print('='*80)

        for opt_level in range(4):

            for num_bits in [
                # 8,
                # 16,
                # 32,
                64
            ]:
                os.makedirs('lib', exist_ok=True)
                save_lib_path = 'lib/{:08d}_{}.so'.format(
                    args.experiment_id if args.experiment_id is not None else 0, num_bits
                )

                compiled_model = CompiledLogicNet(
                    model=model.model,
                    num_bits=num_bits,
                    cpu_compiler='gcc',
                    # cpu_compiler='clang',
                    verbose=True,
                )

                compiled_model.compile(
                    opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
                    save_lib_path=save_lib_path,
                    verbose=True
                )

                correct, total = 0, 0
                with torch.no_grad():
                    for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
                        data = torch.nn.Flatten()(data).bool().numpy()

                        output = compiled_model(data, verbose=True)

                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]

                acc3 = correct / total
                print('COMPILED MODEL', num_bits, acc3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_scale', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=6.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')
    parser.add_argument('-eid', '--experiment_id', type=int, default=None)
    args = parser.parse_args()
    main(args)
