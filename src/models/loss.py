import torch
import torch.nn.functional as F

class LossWrapper(torch.nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self,model_output,targets,has_prediction,step = None):
        return self.loss_fn(model_output, targets, has_prediction, step)


class Type2DirichletLoss(torch.nn.Module):

    def __init__(self,num_classes= 100,reg_weight= 0.001,warmup_steps= 2500,max_reg_weight = 0.05):
        super().__init__()
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.warmup_steps = warmup_steps
        self.max_reg_weight = max_reg_weight
        self.uniform_alpha = torch.ones(num_classes)

    def forward (self,model_output,targets,has_prediction,step= None) :
        logits = model_output.number_logits
        alpha = torch.exp(logits) + 1.0
        S = alpha.sum(dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        ml_loss = torch.sum(targets_one_hot * (torch.log(S.unsqueeze(1)) - torch.log(alpha)), dim=1)
    
        uniform_alpha = self.uniform_alpha.to(alpha.device)
        digamma_alpha = torch.digamma(alpha)
        digamma_S = torch.digamma(S.unsqueeze(1))
        kl_div = (torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1)
                - torch.lgamma(torch.tensor(self.num_classes, device=alpha.device))
                + ((alpha - uniform_alpha) * (digamma_alpha - digamma_S)).sum(dim=1))

        # Compute mean losses with masks
        mean_ml_loss = (has_prediction * ml_loss).sum() / (has_prediction.sum() + 1e-6)
        mean_kl_div = ((1 - has_prediction) * kl_div).sum() / ((1 - has_prediction).sum() + 1e-6)

        # Apply warmup to regularization weight if step is provided
        current_reg_weight = self.reg_weight
        if step is not None:
            progress = min(1.0, step / self.warmup_steps)
            current_reg_weight = self.reg_weight + progress * (self.max_reg_weight - self.reg_weight)
        total_loss = mean_ml_loss + current_reg_weight * mean_kl_div

        return total_loss, mean_ml_loss, mean_kl_div


class SoftmaxWithUncertaintyLoss(torch.nn.Module):
    def __init__(self, num_classes= 100, label_smoothing= 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.total_classes = num_classes + 1

    def forward(self,model_output,targets,has_prediction,step= None,):
        # Use all_logits from model output (including absent class)
        logits = model_output.all_logits
        batch_size = logits.shape[0]

        # Create modified targets where samples without predictions point to the absent class (index 100)
        modified_targets = targets.clone()
        absent_class_idx = self.num_classes
        modified_targets[has_prediction == 0] = absent_class_idx

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(logits,modified_targets,label_smoothing=self.label_smoothing,reduction="none")
        # Compute mean loss
        mean_ce_loss = ce_loss.mean()
        # Return tuple for API compatibility with DirichletLoss
        # The second and third values are the same since we don't have separate components
        return mean_ce_loss, mean_ce_loss, torch.tensor(0.0, device=logits.device)