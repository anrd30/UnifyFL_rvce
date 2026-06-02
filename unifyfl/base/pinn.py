"""
Adversarial PINN Guard implementation for UnifyFL.
Enforces physical manifold constraints (Laplacian residual) on model logit outputs.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional

class PINNGuard(nn.Module):
    """PINN Guard MLP architecture."""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_layers: int = 3, activation: str = 'tanh'):
        super().__init__()
        act_fn = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
        }[activation.lower()]
        
        layers = []
        prev_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AdversarialAttacker(nn.Module):
    """Generator to produce adversarial logit perturbations."""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, clean_logits: torch.Tensor) -> torch.Tensor:
        perturbation = self.net(clean_logits)
        return clean_logits + perturbation


class FisherInformationMetric:
    """
    Fisher Information Metric on the probability simplex.

    Instead of treating logits in flat Euclidean space, we map them to the
    probability simplex via softmax and weight gradients by the Fisher
    Information Metric (the natural Riemannian metric on categorical
    distributions). The metric tensor for a categorical distribution p is
    diagonal: g_ij = δ_ij / p_i. This gives the simplex non-trivial curvature,
    amplifying perturbations in low-probability regions where backdoors hide.
    """

    @staticmethod
    def fisher_weights(logits: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Return the diagonal Fisher weights g^{ii} = 1/p_i for given logits."""
        probs = torch.softmax(logits, dim=-1).clamp(min=eps)
        return 1.0 / probs


def _compute_physics_loss(
    model: nn.Module,
    logits: torch.Tensor,
    detach_input: bool = True,
    use_fisher: bool = False,
) -> torch.Tensor:
    """
    Compute the Laplacian residual ||∇²f(x)||² for the PINN Guard.

    If use_fisher is True, the Laplacian is weighted by the Fisher Information
    Metric on the input simplex (1/p_i), making the curvature measure genuinely
    non-Euclidean rather than flat. Training and scoring must use the same
    use_fisher value, or the scores won't match the learned objective.
    """
    if detach_input:
        x = logits.clone().detach().requires_grad_(True)
    else:
        x = logits if logits.requires_grad else logits.requires_grad_(True)

    output = model(x)
    B, C = output.shape
    laplacian = torch.zeros_like(output)

    for i in range(C):
        grad_i = torch.autograd.grad(
            output[:, i].sum(), x,
            create_graph=True, retain_graph=True,
        )[0]

        for j in range(C):
            grad_ij = torch.autograd.grad(
                grad_i[:, j].sum(), x,
                create_graph=True, retain_graph=True,
            )[0]
            laplacian[:, i] = laplacian[:, i] + grad_ij[:, j]

    # Weight the curvature by the Fisher metric on the input space (g^{ij} = 1/p_i)
    if use_fisher:
        laplacian = laplacian * FisherInformationMetric.fisher_weights(x)

    return (laplacian ** 2).mean()

def train_adversarial_pinn_guard(
    clean_logits: torch.Tensor,
    target: int = 0,
    n_epochs: int = 100,
    n_inner_pinn: int = 3,
    n_inner_adv: int = 2,
    pinn_lr: float = 1e-3,
    adv_lr: float = 1e-3,
    attack_weight: float = 1.0,
    evasion_weight: float = 10.0,
    pinn_config: Optional[Dict] = None,
    device: str = 'cpu',
    verbose: bool = False,
    use_fisher: bool = False,
) -> Tuple[PINNGuard, dict]:
    """
    Train Adversarial Min-Max PINN Guard on clean logits.

    use_fisher: if True, the physics loss is weighted by the Fisher Information
    Metric on the input simplex. The scorer must use the same flag at inference.
    """
    C = clean_logits.shape[1]
    clean_logits = clean_logits.to(device)
    
    if pinn_config is None:
        pinn_config = {'input_dim': C, 'hidden_dim': 64, 'num_layers': 3, 'activation': 'tanh'}
    pinn = PINNGuard(**pinn_config).to(device)
    adversary = AdversarialAttacker(input_dim=C, hidden_dim=64).to(device)
    
    pinn_optimizer = optim.Adam(pinn.parameters(), lr=pinn_lr)
    adv_optimizer = optim.Adam(adversary.parameters(), lr=adv_lr)
    
    history = {
        'pinn_loss': [], 'adv_loss': [],
        'clean_violation': [], 'adv_violation': [],
    }
    
    for epoch in range(n_epochs):
        pinn.train()
        
        # Train PINN
        pinn_loss_epoch = 0.0
        for _ in range(n_inner_pinn):
            pinn_optimizer.zero_grad()
            clean_energy = _compute_physics_loss(pinn, clean_logits, detach_input=True, use_fisher=use_fisher)

            with torch.no_grad():
                adv_logits = adversary(clean_logits)
            adv_energy = _compute_physics_loss(pinn, adv_logits, detach_input=True, use_fisher=use_fisher)
            
            pinn_loss = clean_energy - 0.5 * adv_energy
            pinn_loss.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
            pinn_optimizer.step()
            pinn_loss_epoch = pinn_loss.item()
            
        # Train Adversary
        adv_loss_epoch = 0.0
        for _ in range(n_inner_adv):
            adv_optimizer.zero_grad()
            adv_logits = adversary(clean_logits)
            
            evasion_loss = _compute_physics_loss(pinn, adv_logits, detach_input=False, use_fisher=use_fisher)
            utility = -adv_logits[:, target].mean()
            distance = torch.norm(adv_logits - clean_logits, dim=-1).mean()
            
            adv_loss = evasion_weight * evasion_loss + attack_weight * utility + 5.0 * distance
            adv_loss.backward()
            torch.nn.utils.clip_grad_norm_(adversary.parameters(), max_norm=1.0)
            adv_optimizer.step()
            adv_loss_epoch = adv_loss.item()
        
        # Record history
        history['pinn_loss'].append(pinn_loss_epoch)
        history['adv_loss'].append(adv_loss_epoch)
        
        if verbose and (epoch + 1) % 20 == 0:
            with torch.no_grad():
                clean_v = pinn(clean_logits).abs().mean().item()
            print(f"Epoch {epoch+1}/{n_epochs} | PINN Loss: {pinn_loss_epoch:.6f} | Clean V: {clean_v:.6f}")
            
    pinn.eval()
    return pinn, history

if __name__ == "__main__":
    print("Testing PINN Guard initialization...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clean = torch.randn(100, 10)
    pinn, _ = train_adversarial_pinn_guard(clean, device=device, n_epochs=10, verbose=True)
    score = _compute_physics_loss(pinn, clean.to(device)).item()
    print(f"Sanity test complete. Clean logit score: {score:.6f}")
