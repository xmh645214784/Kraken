import torch
from .optimizer import Optimizer


class RAdagrad(Optimizer):
    r"""

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.9999, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RAdagrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdagrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # if grad.is_sparse:
                #     print(grad.coalesce())
                #     print(p.data)
                #     raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if grad.is_sparse:
                        raise RuntimeError('weight_decay option is not compatible with sparse gradients in RMSprop')
                    grad = grad.add(group['weight_decay'], p.data)

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()
                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)
                    length = 1
                    for i in grad_values.size():
                        if i == 0:
                            raise RuntimeError()
                        length *= i
                    grad_v = grad_values.clone().detach().pow_(2)
                    grad_v = grad_v.sum() / length
                    grad_v = torch.full(grad_values.size(), grad_v).to(grad.device)
                    grad_2 = make_sparse(grad_v)
                    square_avg.mul_(alpha).add_((1 - alpha), grad_2)
                    square_avg = square_avg.sparse_mask(grad)
                    if group['centered']:
                        grad_avg = state['grad_avg']
                        grad_avg.mul_(alpha).add_(1 - alpha, grad_values)
                        grad_avg_2 = make_sparse(grad_avg.pow(2))
                        avg = square_avg.add(-1 * grad_avg_2).sqrt_().add_(group['eps'])
                    else:
                        avg = square_avg._values().sqrt_().add_(group['eps'])

                    if group['momentum'] > 0:
                        buf = state['momentum_buffer']
                        temp = make_sparse(grad_values / avg)
                        buf.mul_(group['momentum']).add_(temp)
                        p.data.add_(-group['lr'], buf)
                    else:
                        temp = make_sparse(grad_values / avg)
                        p.data.add_(-group['lr'], temp)
                else:
                    square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                    if group['centered']:
                        grad_avg = state['grad_avg']
                        grad_avg.mul_(alpha).add_(1 - alpha, grad)
                        avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt_().add_(group['eps'])
                    else:
                        avg = square_avg.sqrt().add_(group['eps'])

                    if group['momentum'] > 0:
                        buf = state['momentum_buffer']
                        buf.mul_(group['momentum']).addcdiv_(grad, avg)
                        p.data.add_(-group['lr'], buf)
                    else:
                        p.data.addcdiv_(-group['lr'], grad, avg)

        return loss

