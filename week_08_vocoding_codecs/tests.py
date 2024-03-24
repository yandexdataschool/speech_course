import torch
import torch.nn as nn

from model import MNISTEncoderDecoder
from vector_quantization import VectorQuantizer, VectorQuantizationLoss


def test_vector_quantization():
    embedding_dim=16
    codebook_size=10
    batch = 8

    vq =  VectorQuantizer(embedding_dim=embedding_dim, codebook_size=codebook_size)
    weight = nn.Parameter(torch.arange(codebook_size).unsqueeze(dim=1).repeat(1, embedding_dim).float())
    vq.codebook.weight = weight

    inp_small = -10. + torch.rand(batch, embedding_dim)
    inp_small = inp_small.unsqueeze(dim=2).unsqueeze(dim=3)
    inp_medium = torch.stack([idx * torch.ones(embedding_dim) for idx in range(batch)], dim=0)
    inp_medium = inp_medium.unsqueeze(dim=2).unsqueeze(dim=3)
    inp_big = codebook_size + 10. + torch.rand(batch, embedding_dim)
    inp_big = inp_big.unsqueeze(dim=2).unsqueeze(dim=3)

    assert inp_small.shape == inp_medium.shape == inp_big.shape

    indices_small = vq.encode(inp_small).squeeze(dim=2).squeeze(dim=1)
    indices_medium = vq.encode(inp_medium).squeeze(dim=2).squeeze(dim=1)
    indices_big = vq.encode(inp_big).squeeze(dim=2).squeeze(dim=1)

    assert torch.eq(indices_small, torch.zeros(batch)).all()
    assert torch.eq(indices_medium, torch.arange(batch)).all()
    assert torch.eq(indices_big, codebook_size - 1 + torch.zeros(batch)).all()

    return True


def test_vector_quantisation_loss():
    B, E, H, W = 2, 2, 1, 1

    loss_fn = VectorQuantizationLoss(commitment_cost=1.)

    inputs = torch.ones([B, E, H, W], dtype=torch.float, requires_grad=True)
    quantized = torch.zeros([B, E, H, W], dtype=torch.float, requires_grad=True)

    loss = loss_fn(inputs, quantized)
    loss.backward()

    assert torch.allclose(inputs.grad, 0.5 * torch.ones([B, E, H, W])), \
        "Loss shoulds be based on MSE Loss"
    assert torch.allclose(quantized.grad, - 0.5 * torch.ones([B, E, H, W])), \
        "Loss shoulds be based on MSE Loss"

    for commitment_cost in (0.25, 1., 2.):
        B, E, H, W = 5, 4, 3, 2
        loss_fn = VectorQuantizationLoss(commitment_cost=commitment_cost)

        inputs = torch.randn([B, E, H, W], dtype=torch.float, requires_grad=True)
        quantized = torch.randn([B, E, H, W], dtype=torch.float, requires_grad=True)

        loss = loss_fn(inputs, quantized)
        loss.backward()

        close = torch.allclose(
            input= - inputs.grad,
            other=quantized.grad * commitment_cost)
    
        if not close:
            raise AssertionError(
                "Gradients are supposed to of opposite signs."
                "And differ in commitment cost times")
    return True


def get_gradients_state(module: nn.Module):
    gradients = []
    for p in module.parameters():
        gradients.append(p.grad is not None)
    
    if len(gradients) == 0:
        return "none"
    if sum(gradients) == 0:
        return "no"
    if sum(gradients) == len(gradients):
        return "yes"
    else:
        return "partial"


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, quantied):
        return torch.tensor(0.)


def test_training_step():
    C, E = 12, 16
    B, H, W = 2, 28, 28

    # Reconstruction only loss

    quantizer = VectorQuantizer(codebook_size=C, embedding_dim=E)
    dummy_loss_fn = DummyLoss()

    module = MNISTEncoderDecoder(quantizer=quantizer, vq_loss=dummy_loss_fn)

    batch = torch.randn([B, 1, H, W])

    loss = module.training_step_with_quantizer(batch)
    loss.backward()

    enc_grad = get_gradients_state(module.encoder)
    quant_grad = get_gradients_state(module.quantizer)
    dec_grad = get_gradients_state(module.decoder)

    if (enc_grad, quant_grad, dec_grad) != ("yes", "no", "yes"):
        raise AssertionError(
            "Gradient from reconstruction loss should propagate through encoder and decoder.\n"
            f"But your gradients propagate through encoder={enc_grad} quantizer={quant_grad} decoder={dec_grad}"
        )
    
    # VQ loss only

    quantizer = VectorQuantizer(codebook_size=C, embedding_dim=E)
    vq_loss_fn = VectorQuantizationLoss()

    module = MNISTEncoderDecoder(quantizer=quantizer, vq_loss=vq_loss_fn)
    module.reconstr_loss_fn = DummyLoss()

    batch = torch.randn([B, 1, H, W])

    loss = module.training_step_with_quantizer(batch)
    loss.backward()

    enc_grad = get_gradients_state(module.encoder)
    quant_grad = get_gradients_state(module.quantizer)
    dec_grad = get_gradients_state(module.decoder)

    if (enc_grad, quant_grad, dec_grad) != ("yes", "yes", "no"):
        raise AssertionError(
            "Gradient from vector quatisation loss should propagate through encoder and quantizer.\n"
            f"But your gradients propagate through encoder={enc_grad} quantizer={quant_grad} decoder={dec_grad}"
        )
    
    return True