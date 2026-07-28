import torch
import torch.nn as nn
from src.config import Config
from src.model.base import BaseDeepakeDetectionModel
from src.optimizer.sam import SAM


class MockModel(BaseDeepakeDetectionModel):

    def __init__(self, config: Config):
        super().__init__(config)
        self.layer = nn.Linear(5, 2)

    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        import torch.optim as optim
        from src import config as C

        if self.config.optimizer == C.Optimizer.AdamW:
            optimizer = optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
        elif self.config.optimizer in (C.Optimizer.SAM_SGD, C.Optimizer.SAM_AdamW):
            base_opt_cls = (
                optim.SGD
                if self.config.optimizer == C.Optimizer.SAM_SGD
                else optim.AdamW
            )
            base_opt_kwargs = {
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
            }
            if self.config.optimizer == C.Optimizer.SAM_SGD:
                base_opt_kwargs["momentum"] = self.config.betas[0]
            else:
                base_opt_kwargs["betas"] = self.config.betas

            optimizer = SAM(
                optimizer_grouped_parameters,
                base_optimizer=base_opt_cls,
                rho=self.config.sam_rho,
                adaptive=self.config.sam_adaptive,
                **base_opt_kwargs,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        return {"optimizer": optimizer}


def test_sam_pure_pytorch():
    print("Testing SAM in pure PyTorch...")
    model = nn.Linear(10, 2)
    optimizer = SAM(
        model.parameters(),
        base_optimizer=torch.optim.SGD,
        rho=0.05,
        adaptive=False,
        lr=0.1,
        momentum=0.9,
    )

    # Initial state
    inputs = torch.randn(5, 10)
    targets = torch.randint(0, 2, (5,))
    criterion = nn.CrossEntropyLoss()

    # Step 1: save original parameters
    original_weight = model.weight.clone()

    # Step 2: forward-backward first step
    loss = criterion(model(inputs), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # Weights should be perturbed now
    perturbed_weight = model.weight.clone()
    assert not torch.allclose(
        original_weight, perturbed_weight
    ), "Weights did not perturb after first step"

    # Step 3: forward-backward second step
    loss2 = criterion(model(inputs), targets)
    loss2.backward()
    optimizer.second_step(zero_grad=True)

    # Weights should be restored and updated now
    updated_weight = model.weight.clone()
    assert not torch.allclose(
        perturbed_weight, updated_weight
    ), "Weights did not update/restore after second step"
    print("Pure PyTorch SAM test passed!")


def test_mock_model_optimization():
    print("Testing MockModel with SAM configure_optimizers and optimizer_step...")
    config = Config(
        optimizer="SAM-AdamW",
        sam_rho=0.08,
        sam_adaptive=True,
        run_name="test-sam-tmp",
    )
    model = MockModel(config)

    # Check configure_optimizers
    optimizers = model.configure_optimizers()
    optimizer = optimizers["optimizer"]
    assert isinstance(optimizer, SAM), "Expected SAM optimizer"
    assert (
        optimizer.defaults["rho"] == 0.08
    ), f"Expected rho=0.08, got {optimizer.defaults['rho']}"
    assert optimizer.defaults["adaptive"] is True, "Expected adaptive=True"

    # Check optimizer_step double-pass execution
    inputs = torch.randn(2, 5)
    targets = torch.randint(0, 2, (2,))
    criterion = nn.CrossEntropyLoss()

    original_weights = model.layer.weight.clone()

    # Define the closure that PyTorch Lightning would pass
    def closure():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        return loss

    # Execute optimizer_step
    model.optimizer_step(
        epoch=0,
        batch_idx=0,
        optimizer=optimizer,
        optimizer_closure=closure,
    )

    # Weights should be restored and updated
    updated_weights = model.layer.weight.clone()
    assert not torch.allclose(
        original_weights, updated_weights
    ), "Weights did not update via optimizer_step!"
    print("MockModel SAM optimization test passed!")


if __name__ == "__main__":
    test_sam_pure_pytorch()
    test_mock_model_optimization()
