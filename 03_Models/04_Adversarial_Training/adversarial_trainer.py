"""
Adversarial Training Module for P22 IDS

This module implements adversarial training techniques to improve model robustness
against evasion attacks, targeting the ≥90% detection rate KPI for adversarial samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AdversarialConfig:
    """Configuration for adversarial training."""
    
    # Attack parameters
    attack_type: str = "fgsm"  # fgsm, pgd, c_w
    epsilon: float = 0.01
    alpha: float = 0.001
    num_steps: int = 10
    
    # Training parameters
    adversarial_ratio: float = 0.5  # Ratio of adversarial samples
    lambda_adv: float = 1.0  # Adversarial loss weight
    lambda_clean: float = 1.0  # Clean loss weight
    
    # Robustness parameters
    targeted: bool = False
    random_start: bool = True
    clip_min: float = 0.0
    clip_max: float = 1.0


class AdversarialAttack(ABC):
    """Abstract base class for adversarial attacks."""
    
    def __init__(self, model: nn.Module, config: AdversarialConfig):
        """
        Initialize adversarial attack.
        
        Args:
            model: Target model
            config: Attack configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples.
        
        Args:
            x: Input samples
            y: True labels
            
        Returns:
            Adversarial examples
        """
        pass
    
    def _clip_perturbation(self, x_orig: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """Clip perturbation to epsilon ball."""
        delta = x_adv - x_orig
        delta = torch.clamp(delta, -self.config.epsilon, self.config.epsilon)
        x_adv_clipped = x_orig + delta
        x_adv_clipped = torch.clamp(x_adv_clipped, self.config.clip_min, self.config.clip_max)
        return x_adv_clipped


class FGSMAttack(AdversarialAttack):
    """Fast Gradient Sign Method (FGSM) attack."""
    
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate FGSM adversarial examples."""
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x_adv)
        
        # Calculate loss
        if self.config.targeted:
            # For targeted attacks, minimize loss for target class
            loss = -F.cross_entropy(outputs, y)
        else:
            # For untargeted attacks, maximize loss
            loss = F.cross_entropy(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Generate adversarial example
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + self.config.epsilon * grad_sign
        
        # Clip to valid range
        x_adv = self._clip_perturbation(x, x_adv)
        
        return x_adv.detach()


class PGDAttack(AdversarialAttack):
    """Projected Gradient Descent (PGD) attack."""
    
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate PGD adversarial examples."""
        x_adv = x.clone().detach()
        
        # Random initialization
        if self.config.random_start:
            noise = torch.empty_like(x_adv).uniform_(-self.config.epsilon, self.config.epsilon)
            x_adv = x_adv + noise
            x_adv = self._clip_perturbation(x, x_adv)
        
        # Iterative attack
        for _ in range(self.config.num_steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(x_adv)
            
            # Calculate loss
            if self.config.targeted:
                loss = -F.cross_entropy(outputs, y)
            else:
                loss = F.cross_entropy(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Update adversarial example
            grad = x_adv.grad.detach()
            x_adv = x_adv + self.config.alpha * grad.sign()
            
            # Project back to epsilon ball
            x_adv = self._clip_perturbation(x, x_adv)
            x_adv = x_adv.detach()
        
        return x_adv


class CWAttack(AdversarialAttack):
    """Carlini & Wagner (C&W) attack."""
    
    def __init__(self, model: nn.Module, config: AdversarialConfig):
        super().__init__(model, config)
        self.c = 1.0  # Regularization parameter
        self.kappa = 0.0  # Confidence parameter
    
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate C&W adversarial examples."""
        batch_size = x.size(0)
        
        # Initialize perturbation in tanh space
        w = torch.zeros_like(x, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=0.01)
        
        best_adv = x.clone()
        best_distance = float('inf') * torch.ones(batch_size)
        
        for iteration in range(self.config.num_steps):
            # Convert from tanh space to input space
            x_adv = 0.5 * (torch.tanh(w) + 1) * (self.config.clip_max - self.config.clip_min) + self.config.clip_min
            
            # Forward pass
            outputs = self.model(x_adv)
            
            # C&W loss function
            real_logits = torch.sum(outputs * F.one_hot(y, outputs.size(1)), dim=1)
            other_logits = torch.max((1 - F.one_hot(y, outputs.size(1))) * outputs - 
                                   F.one_hot(y, outputs.size(1)) * 1e4, dim=1)[0]
            
            if self.config.targeted:
                loss1 = torch.clamp(other_logits - real_logits, min=-self.kappa)
            else:
                loss1 = torch.clamp(real_logits - other_logits, min=-self.kappa)
            
            # L2 distance loss
            loss2 = torch.sum((x_adv - x).view(batch_size, -1) ** 2, dim=1)
            
            # Combined loss
            loss = torch.sum(self.c * loss1 + loss2)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update best adversarial examples
            pred_labels = torch.argmax(outputs, dim=1)
            successful = (pred_labels != y) if not self.config.targeted else (pred_labels == y)
            
            distance = torch.sum((x_adv - x).view(batch_size, -1) ** 2, dim=1)
            
            for i in range(batch_size):
                if successful[i] and distance[i] < best_distance[i]:
                    best_distance[i] = distance[i]
                    best_adv[i] = x_adv[i].clone()
        
        return best_adv.detach()


class AdversarialTrainer:
    """
    Adversarial training implementation for robust model training.
    
    This trainer implements various adversarial training strategies to improve
    model robustness against evasion attacks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: AdversarialConfig,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize adversarial trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            config: Adversarial training configuration
            device: Training device
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize attack method
        self.attack = self._create_attack()
        
        # Training statistics
        self.stats = {
            'clean_accuracy': [],
            'adversarial_accuracy': [],
            'clean_loss': [],
            'adversarial_loss': []
        }
    
    def _create_attack(self) -> AdversarialAttack:
        """Create attack method based on configuration."""
        attack_map = {
            'fgsm': FGSMAttack,
            'pgd': PGDAttack,
            'c_w': CWAttack
        }
        
        if self.config.attack_type not in attack_map:
            raise ValueError(f"Unknown attack type: {self.config.attack_type}")
        
        return attack_map[self.config.attack_type](self.model, self.config)
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Perform one adversarial training step.
        
        Args:
            x: Input batch
            y: Target labels
            
        Returns:
            Training statistics
        """
        self.model.train()
        
        # Move to device
        x, y = x.to(self.device), y.to(self.device)
        
        # Generate adversarial examples
        batch_size = x.size(0)
        num_adv = int(batch_size * self.config.adversarial_ratio)
        
        if num_adv > 0:
            # Split batch into clean and adversarial
            x_clean = x[num_adv:]
            y_clean = y[num_adv:]
            
            x_for_adv = x[:num_adv]
            y_for_adv = y[:num_adv]
            
            # Generate adversarial examples
            with torch.no_grad():
                x_adv = self.attack.generate(x_for_adv, y_for_adv)
            
            # Combine clean and adversarial samples
            x_combined = torch.cat([x_clean, x_adv], dim=0)
            y_combined = torch.cat([y_clean, y_for_adv], dim=0)
        else:
            x_combined = x
            y_combined = y
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(x_combined)
        
        # Calculate losses
        if num_adv > 0:
            # Separate clean and adversarial losses
            clean_outputs = outputs[:len(x_clean)]
            adv_outputs = outputs[len(x_clean):]
            
            clean_loss = F.cross_entropy(clean_outputs, y_clean) if len(x_clean) > 0 else 0
            adv_loss = F.cross_entropy(adv_outputs, y_for_adv)
            
            # Combined loss
            total_loss = (self.config.lambda_clean * clean_loss + 
                         self.config.lambda_adv * adv_loss)
        else:
            clean_loss = F.cross_entropy(outputs, y_combined)
            adv_loss = torch.tensor(0.0)
            total_loss = clean_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Calculate accuracies
        with torch.no_grad():
            if num_adv > 0:
                clean_acc = (torch.argmax(clean_outputs, dim=1) == y_clean).float().mean().item() if len(x_clean) > 0 else 0
                adv_acc = (torch.argmax(adv_outputs, dim=1) == y_for_adv).float().mean().item()
            else:
                clean_acc = (torch.argmax(outputs, dim=1) == y_combined).float().mean().item()
                adv_acc = 0.0
        
        # Update statistics
        step_stats = {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'clean_loss': clean_loss.item() if isinstance(clean_loss, torch.Tensor) else clean_loss,
            'adversarial_loss': adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            'total_loss': total_loss.item()
        }
        
        return step_stats
    
    def evaluate_robustness(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model robustness against adversarial attacks.
        
        Args:
            x: Input samples
            y: True labels
            
        Returns:
            Robustness metrics
        """
        self.model.eval()
        x, y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            # Clean accuracy
            clean_outputs = self.model(x)
            clean_acc = (torch.argmax(clean_outputs, dim=1) == y).float().mean().item()
        
        # Generate adversarial examples
        x_adv = self.attack.generate(x, y)
        
        with torch.no_grad():
            # Adversarial accuracy
            adv_outputs = self.model(x_adv)
            adv_acc = (torch.argmax(adv_outputs, dim=1) == y).float().mean().item()
        
        # Calculate perturbation statistics
        perturbation = torch.norm((x_adv - x).view(x.size(0), -1), p=2, dim=1)
        avg_perturbation = perturbation.mean().item()
        max_perturbation = perturbation.max().item()
        
        return {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'robustness_gap': clean_acc - adv_acc,
            'avg_perturbation': avg_perturbation,
            'max_perturbation': max_perturbation
        }
    
    def adaptive_epsilon_scheduling(self, epoch: int, total_epochs: int) -> None:
        """
        Adaptive epsilon scheduling for curriculum adversarial training.
        
        Args:
            epoch: Current epoch
            total_epochs: Total training epochs
        """
        # Linear increase in epsilon
        progress = epoch / total_epochs
        max_epsilon = self.config.epsilon
        current_epsilon = max_epsilon * progress
        
        self.config.epsilon = current_epsilon
        self.attack.config.epsilon = current_epsilon
        
        self.logger.info(f"Epoch {epoch}: Updated epsilon to {current_epsilon:.4f}")


class RobustnessEvaluator:
    """Comprehensive robustness evaluation against multiple attack types."""
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device('cpu')):
        """
        Initialize robustness evaluator.
        
        Args:
            model: Model to evaluate
            device: Evaluation device
        """
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_evaluation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attack_configs: List[AdversarialConfig]
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive robustness evaluation.
        
        Args:
            x: Input samples
            y: True labels
            attack_configs: List of attack configurations to test
            
        Returns:
            Evaluation results for each attack type
        """
        results = {}
        
        # Clean accuracy baseline
        self.model.eval()
        x, y = x.to(self.device), y.to(self.device)
        
        with torch.no_grad():
            clean_outputs = self.model(x)
            clean_accuracy = (torch.argmax(clean_outputs, dim=1) == y).float().mean().item()
        
        results['clean'] = {'accuracy': clean_accuracy}
        
        # Test each attack configuration
        for i, config in enumerate(attack_configs):
            attack_name = f"{config.attack_type}_eps_{config.epsilon}"
            
            try:
                # Create attack
                attack_map = {
                    'fgsm': FGSMAttack,
                    'pgd': PGDAttack,
                    'c_w': CWAttack
                }
                
                attack = attack_map[config.attack_type](self.model, config)
                
                # Generate adversarial examples
                x_adv = attack.generate(x, y)
                
                # Evaluate
                with torch.no_grad():
                    adv_outputs = self.model(x_adv)
                    adv_accuracy = (torch.argmax(adv_outputs, dim=1) == y).float().mean().item()
                
                # Calculate metrics
                perturbation = torch.norm((x_adv - x).view(x.size(0), -1), p=2, dim=1)
                
                results[attack_name] = {
                    'accuracy': adv_accuracy,
                    'robustness_gap': clean_accuracy - adv_accuracy,
                    'avg_perturbation': perturbation.mean().item(),
                    'max_perturbation': perturbation.max().item(),
                    'success_rate': 1.0 - adv_accuracy  # Attack success rate
                }
                
                self.logger.info(f"Attack {attack_name}: Accuracy = {adv_accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate attack {attack_name}: {e}")
                results[attack_name] = {'error': str(e)}
        
        return results
    
    def generate_robustness_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """Generate a comprehensive robustness report."""
        
        report = "=" * 60 + "\n"
        report += "ADVERSARIAL ROBUSTNESS EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Clean performance
        if 'clean' in results:
            report += f"Clean Accuracy: {results['clean']['accuracy']:.4f}\n\n"
        
        # Attack results
        report += "Attack Results:\n"
        report += "-" * 40 + "\n"
        
        for attack_name, metrics in results.items():
            if attack_name == 'clean':
                continue
            
            if 'error' in metrics:
                report += f"{attack_name}: ERROR - {metrics['error']}\n"
                continue
            
            report += f"\n{attack_name}:\n"
            report += f"  Accuracy: {metrics['accuracy']:.4f}\n"
            report += f"  Robustness Gap: {metrics['robustness_gap']:.4f}\n"
            report += f"  Attack Success Rate: {metrics['success_rate']:.4f}\n"
            report += f"  Avg Perturbation: {metrics['avg_perturbation']:.6f}\n"
        
        # Overall assessment
        report += "\n" + "=" * 60 + "\n"
        report += "OVERALL ASSESSMENT\n"
        report += "=" * 60 + "\n"
        
        # Calculate average robustness
        valid_attacks = [metrics for name, metrics in results.items() 
                        if name != 'clean' and 'error' not in metrics]
        
        if valid_attacks:
            avg_robustness = np.mean([metrics['accuracy'] for metrics in valid_attacks])
            report += f"Average Adversarial Accuracy: {avg_robustness:.4f}\n"
            
            # P22 KPI assessment
            kpi_threshold = 0.90
            meets_kpi = avg_robustness >= kpi_threshold
            report += f"P22 Robustness KPI (≥90%): {'✓ PASS' if meets_kpi else '✗ FAIL'}\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Example model (placeholder)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 10)
        
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    
    # Initialize components
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    config = AdversarialConfig(
        attack_type="pgd",
        epsilon=0.01,
        num_steps=10,
        adversarial_ratio=0.5
    )
    
    trainer = AdversarialTrainer(model, optimizer, config)
    
    # Example training step
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))
    
    stats = trainer.train_step(x, y)
    print("Training statistics:", stats)
    
    # Example robustness evaluation
    evaluator = RobustnessEvaluator(model)
    
    attack_configs = [
        AdversarialConfig(attack_type="fgsm", epsilon=0.01),
        AdversarialConfig(attack_type="pgd", epsilon=0.01, num_steps=10),
    ]
    
    results = evaluator.comprehensive_evaluation(x, y, attack_configs)
    report = evaluator.generate_robustness_report(results)
    print("\n" + report)
