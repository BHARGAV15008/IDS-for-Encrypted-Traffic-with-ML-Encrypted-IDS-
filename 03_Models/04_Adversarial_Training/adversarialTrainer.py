"""
Adversarial Training Module
Implements adversarial training techniques for robust IDS
Includes FGSM, PGD, and other adversarial attack methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FGSMAttack:
    """Fast Gradient Sign Method (FGSM) adversarial attack"""
    
    def __init__(self, epsilon: float = 0.01):
        """
        Args:
            epsilon: Perturbation magnitude
        """
        self.epsilon = epsilon
        
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        lossFn: nn.Module
    ) -> torch.Tensor:
        """
        Generate FGSM adversarial examples
        
        Args:
            model: Target model
            x: Input samples (batch, ...)
            y: True labels (batch,)
            lossFn: Loss function
            
        Returns:
            Adversarial examples
        """
        # Ensure input requires gradient
        x.requires_grad = True
        
        # Forward pass
        outputs = model(x)
        loss = lossFn(outputs, y)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate perturbation
        perturbation = self.epsilon * x.grad.sign()
        
        # Create adversarial example
        xAdv = x + perturbation
        xAdv = torch.clamp(xAdv, 0, 1)  # Ensure valid range
        
        return xAdv.detach()


class PGDAttack:
    """Projected Gradient Descent (PGD) adversarial attack"""
    
    def __init__(
        self,
        epsilon: float = 0.01,
        alpha: float = 0.002,
        numSteps: int = 10,
        randomStart: bool = True
    ):
        """
        Args:
            epsilon: Maximum perturbation
            alpha: Step size
            numSteps: Number of iterations
            randomStart: Whether to start from random point
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.numSteps = numSteps
        self.randomStart = randomStart
        
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        lossFn: nn.Module
    ) -> torch.Tensor:
        """
        Generate PGD adversarial examples
        
        Args:
            model: Target model
            x: Input samples
            y: True labels
            lossFn: Loss function
            
        Returns:
            Adversarial examples
        """
        xAdv = x.clone().detach()
        
        # Random initialization
        if self.randomStart:
            xAdv = xAdv + torch.empty_like(xAdv).uniform_(-self.epsilon, self.epsilon)
            xAdv = torch.clamp(xAdv, 0, 1)
            
        # Iterative attack
        for _ in range(self.numSteps):
            xAdv.requires_grad = True
            
            outputs = model(xAdv)
            loss = lossFn(outputs, y)
            
            model.zero_grad()
            loss.backward()
            
            # Update adversarial example
            with torch.no_grad():
                perturbation = self.alpha * xAdv.grad.sign()
                xAdv = xAdv + perturbation
                
                # Project back to epsilon ball
                delta = torch.clamp(xAdv - x, -self.epsilon, self.epsilon)
                xAdv = torch.clamp(x + delta, 0, 1)
                
        return xAdv.detach()


class CWAttack:
    """Carlini & Wagner (C&W) attack"""
    
    def __init__(
        self,
        c: float = 1.0,
        kappa: float = 0,
        numSteps: int = 100,
        learningRate: float = 0.01
    ):
        """
        Args:
            c: Confidence parameter
            kappa: Margin parameter
            numSteps: Number of optimization steps
            learningRate: Learning rate
        """
        self.c = c
        self.kappa = kappa
        self.numSteps = numSteps
        self.learningRate = learningRate
        
    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate C&W adversarial examples
        
        Args:
            model: Target model
            x: Input samples
            y: True labels
            
        Returns:
            Adversarial examples
        """
        # Initialize perturbation
        delta = torch.zeros_like(x, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.learningRate)
        
        for _ in range(self.numSteps):
            xAdv = x + delta
            outputs = model(xAdv)
            
            # C&W loss
            realLogits = outputs[torch.arange(len(y)), y]
            otherLogits = outputs.clone()
            otherLogits[torch.arange(len(y)), y] = -float('inf')
            maxOtherLogits = otherLogits.max(dim=1)[0]
            
            loss = torch.clamp(realLogits - maxOtherLogits + self.kappa, min=0).sum()
            loss += self.c * delta.norm()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return (x + delta).detach()


class AdversarialTrainer:
    """
    Adversarial training framework for robust model training
    """
    
    def __init__(
        self,
        model: nn.Module,
        attackMethod: str = 'fgsm',
        epsilon: float = 0.01,
        alpha: float = 0.5,
        **attackKwargs
    ):
        """
        Args:
            model: Model to train
            attackMethod: Attack method ('fgsm', 'pgd', 'cw')
            epsilon: Perturbation magnitude
            alpha: Weight for adversarial loss (0-1)
            **attackKwargs: Additional attack parameters
        """
        self.model = model
        self.attackMethod = attackMethod
        self.alpha = alpha
        
        # Initialize attack
        if attackMethod == 'fgsm':
            self.attack = FGSMAttack(epsilon=epsilon, **attackKwargs)
        elif attackMethod == 'pgd':
            self.attack = PGDAttack(epsilon=epsilon, **attackKwargs)
        elif attackMethod == 'cw':
            self.attack = CWAttack(**attackKwargs)
        else:
            raise ValueError(f"Unknown attack method: {attackMethod}")
            
        logger.info(f"Adversarial Trainer initialized: method={attackMethod}, epsilon={epsilon}")
        
    def trainStep(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        lossFn: nn.Module
    ) -> Dict[str, float]:
        """
        Single adversarial training step
        
        Args:
            x: Input batch
            y: Labels
            optimizer: Optimizer
            lossFn: Loss function
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        
        # Clean loss
        outputsClean = self.model(x)
        lossClean = lossFn(outputsClean, y)
        
        # Generate adversarial examples
        if self.attackMethod == 'cw':
            xAdv = self.attack.generate(self.model, x, y)
        else:
            xAdv = self.attack.generate(self.model, x, y, lossFn)
            
        # Adversarial loss
        outputsAdv = self.model(xAdv)
        lossAdv = lossFn(outputsAdv, y)
        
        # Combined loss
        totalLoss = (1 - self.alpha) * lossClean + self.alpha * lossAdv
        
        # Backward pass
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()
        
        return {
            'total_loss': totalLoss.item(),
            'clean_loss': lossClean.item(),
            'adv_loss': lossAdv.item()
        }
        
    def evaluate(
        self,
        xTest: torch.Tensor,
        yTest: torch.Tensor,
        lossFn: nn.Module
    ) -> Dict[str, float]:
        """
        Evaluate model on clean and adversarial examples
        
        Args:
            xTest: Test inputs
            yTest: Test labels
            lossFn: Loss function
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Clean accuracy
            outputsClean = self.model(xTest)
            lossClean = lossFn(outputsClean, yTest)
            predsClean = outputsClean.argmax(dim=1)
            accClean = (predsClean == yTest).float().mean()
            
        # Adversarial accuracy
        if self.attackMethod == 'cw':
            xAdv = self.attack.generate(self.model, xTest, yTest)
        else:
            xAdv = self.attack.generate(self.model, xTest, yTest, lossFn)
            
        with torch.no_grad():
            outputsAdv = self.model(xAdv)
            lossAdv = lossFn(outputsAdv, yTest)
            predsAdv = outputsAdv.argmax(dim=1)
            accAdv = (predsAdv == yTest).float().mean()
            
        return {
            'clean_loss': lossClean.item(),
            'clean_acc': accClean.item(),
            'adv_loss': lossAdv.item(),
            'adv_acc': accAdv.item(),
            'robustness': accAdv.item() / (accClean.item() + 1e-10)
        }


class MixedAdversarialTrainer(AdversarialTrainer):
    """
    Mixed adversarial training using multiple attack methods
    """
    
    def __init__(
        self,
        model: nn.Module,
        attackMethods: list = ['fgsm', 'pgd'],
        epsilon: float = 0.01,
        alpha: float = 0.5
    ):
        """
        Args:
            model: Model to train
            attackMethods: List of attack methods
            epsilon: Perturbation magnitude
            alpha: Weight for adversarial loss
        """
        self.model = model
        self.alpha = alpha
        
        # Initialize multiple attacks
        self.attacks = {}
        for method in attackMethods:
            if method == 'fgsm':
                self.attacks[method] = FGSMAttack(epsilon=epsilon)
            elif method == 'pgd':
                self.attacks[method] = PGDAttack(epsilon=epsilon)
                
        logger.info(f"Mixed Adversarial Trainer initialized: methods={attackMethods}")
        
    def trainStep(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        lossFn: nn.Module
    ) -> Dict[str, float]:
        """Training step with mixed attacks"""
        self.model.train()
        
        # Clean loss
        outputsClean = self.model(x)
        lossClean = lossFn(outputsClean, y)
        
        # Generate adversarial examples from each attack
        advLosses = []
        for attackName, attack in self.attacks.items():
            xAdv = attack.generate(self.model, x, y, lossFn)
            outputsAdv = self.model(xAdv)
            lossAdv = lossFn(outputsAdv, y)
            advLosses.append(lossAdv)
            
        # Average adversarial loss
        lossAdvAvg = torch.stack(advLosses).mean()
        
        # Combined loss
        totalLoss = (1 - self.alpha) * lossClean + self.alpha * lossAdvAvg
        
        # Backward pass
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()
        
        return {
            'total_loss': totalLoss.item(),
            'clean_loss': lossClean.item(),
            'adv_loss': lossAdvAvg.item()
        }


class AdversarialRegularizer:
    """
    Adversarial regularization for training
    Adds adversarial perturbations as regularization term
    """
    
    def __init__(
        self,
        epsilon: float = 0.01,
        regWeight: float = 0.1
    ):
        """
        Args:
            epsilon: Perturbation magnitude
            regWeight: Regularization weight
        """
        self.epsilon = epsilon
        self.regWeight = regWeight
        
    def computeRegularization(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        lossFn: nn.Module
    ) -> torch.Tensor:
        """
        Compute adversarial regularization term
        
        Args:
            model: Model
            x: Input
            y: Labels
            lossFn: Loss function
            
        Returns:
            Regularization loss
        """
        # Generate small perturbation
        x.requires_grad = True
        outputs = model(x)
        loss = lossFn(outputs, y)
        
        model.zero_grad()
        loss.backward()
        
        # Virtual adversarial perturbation
        perturbation = self.epsilon * x.grad / (x.grad.norm() + 1e-10)
        
        # Compute regularization
        xPerturbed = x + perturbation
        outputsPerturbed = model(xPerturbed)
        
        # KL divergence between clean and perturbed
        regLoss = F.kl_div(
            F.log_softmax(outputsPerturbed, dim=1),
            F.softmax(outputs.detach(), dim=1),
            reduction='batchmean'
        )
        
        return self.regWeight * regLoss


class RobustnessEvaluator:
    """
    Evaluate model robustness against various attacks
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: Model to evaluate
        """
        self.model = model
        
    def evaluateRobustness(
        self,
        xTest: torch.Tensor,
        yTest: torch.Tensor,
        epsilons: list = [0.001, 0.005, 0.01, 0.05, 0.1]
    ) -> Dict[str, list]:
        """
        Evaluate robustness across different epsilon values
        
        Args:
            xTest: Test inputs
            yTest: Test labels
            epsilons: List of epsilon values to test
            
        Returns:
            Dictionary of accuracies for each epsilon
        """
        results = {
            'epsilon': epsilons,
            'fgsm_acc': [],
            'pgd_acc': []
        }
        
        lossFn = nn.CrossEntropyLoss()
        
        for eps in epsilons:
            logger.info(f"Evaluating robustness at epsilon={eps}")
            
            # FGSM attack
            fgsmAttack = FGSMAttack(epsilon=eps)
            xAdvFgsm = fgsmAttack.generate(self.model, xTest, yTest, lossFn)
            
            with torch.no_grad():
                outputsFgsm = self.model(xAdvFgsm)
                predsFgsm = outputsFgsm.argmax(dim=1)
                accFgsm = (predsFgsm == yTest).float().mean().item()
                
            results['fgsm_acc'].append(accFgsm)
            
            # PGD attack
            pgdAttack = PGDAttack(epsilon=eps)
            xAdvPgd = pgdAttack.generate(self.model, xTest, yTest, lossFn)
            
            with torch.no_grad():
                outputsPgd = self.model(xAdvPgd)
                predsPgd = outputsPgd.argmax(dim=1)
                accPgd = (predsPgd == yTest).float().mean().item()
                
            results['pgd_acc'].append(accPgd)
            
        return results
    
    def computeRobustnessScore(
        self,
        xTest: torch.Tensor,
        yTest: torch.Tensor,
        epsilon: float = 0.01
    ) -> float:
        """
        Compute overall robustness score
        
        Args:
            xTest: Test inputs
            yTest: Test labels
            epsilon: Perturbation magnitude
            
        Returns:
            Robustness score (0-1)
        """
        results = self.evaluateRobustness(xTest, yTest, [epsilon])
        
        # Average of FGSM and PGD accuracies
        robustnessScore = (results['fgsm_acc'][0] + results['pgd_acc'][0]) / 2
        
        return robustnessScore


class DefensiveDistillation:
    """
    Defensive distillation for adversarial robustness
    """
    
    def __init__(
        self,
        teacherModel: nn.Module,
        studentModel: nn.Module,
        temperature: float = 10.0
    ):
        """
        Args:
            teacherModel: Teacher model
            studentModel: Student model
            temperature: Distillation temperature
        """
        self.teacherModel = teacherModel
        self.studentModel = studentModel
        self.temperature = temperature
        
    def distill(
        self,
        xTrain: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        numEpochs: int = 10
    ):
        """
        Perform defensive distillation
        
        Args:
            xTrain: Training data
            optimizer: Optimizer for student
            numEpochs: Number of training epochs
        """
        self.teacherModel.eval()
        
        for epoch in range(numEpochs):
            self.studentModel.train()
            
            # Get soft labels from teacher
            with torch.no_grad():
                teacherOutputs = self.teacherModel(xTrain) / self.temperature
                softLabels = F.softmax(teacherOutputs, dim=1)
                
            # Train student
            studentOutputs = self.studentModel(xTrain) / self.temperature
            loss = F.kl_div(
                F.log_softmax(studentOutputs, dim=1),
                softLabels,
                reduction='batchmean'
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.info(f"Distillation epoch {epoch+1}/{numEpochs}, loss={loss.item():.4f}")
