from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SubDecoder(nn.Module):
    def __init__(self, d_model, n_codes, n_codebooks):
        super().__init__()
        self.n_codes, self.n_codebooks = n_codes, n_codebooks
        self.d_model = d_model

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.decoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=2)
        self.out_proj = nn.Linear(in_features=d_model, out_features=n_codes)

        self.codes_embedder = nn.Embedding(num_embeddings=n_codes, embedding_dim=d_model)
        self.positional_encoding = nn.Embedding(num_embeddings=n_codebooks, embedding_dim=d_model)

        self.codes_positional_encoding = nn.Embedding(num_embeddings=n_codebooks + 1, embedding_dim=d_model)

    def forward(
        self,
        emb_sequence, # [B, L, d]
        codes_sequence, # [B, L, N]
    ):
        """
        emb_sequence: FloatTensor of size [batch, codes_len, d_model]
        codes_sequence: LongTensor of size [batch, codes_len, n_codebooks]
        return: logits FloatTensor of size [batch, codes_len, n_codes, n_codes_in_codebooks]

        This method uses emb_sequence from encoder-decoder to predict codes logits.
        It applies a decoder-only tranformer in an autoregressive manner and makes exactly n_codebooks steps.
        """
        device = emb_sequence.device

        # Calculate the embeddings of each token in codec
        codes_embeddings = self.codes_embedder(codes_sequence) # [B, L, N, d]
        B, L, N, d = codes_embeddings.shape

        # Flattening batch and codes_len dimensions, for proper usage in transformer decoder
        emb_sequence = emb_sequence.view(B * L, 1, d)
        codes_embeddings = codes_embeddings.view(B * L, N, d)

        # Simple positional encoding
        codes_range = torch.arange(N + 1, device=device).unsqueeze(dim=0)
        codes_PE = self.codes_positional_encoding(codes_range)

        # Add emb_sequence to the beginning of each token_sequence
        src = torch.cat([emb_sequence, codes_embeddings], dim=1)
        src = src + codes_PE

        # Calculate the embeddings of each token in codec
        embeddings = self.decoder(
            src=src,
            mask=nn.Transformer.generate_square_subsequent_mask(N + 1, device=device)
        )
        # Projecting embedding to the number of codes
        # You can further use softmax to get the probabilities of each code, but not inside this function
        embeddings = self.out_proj(embeddings)
        return embeddings.view(B, L, N + 1, self.n_codes)

    @torch.no_grad()
    def autoregressive_sampling(
        self,
        embedding, # [B, d]
        sampling_fn: Callable = lambda x: x.argmax(dim=-1),
    ):
        """
        embedding: FloatTensor of size [batch, d_model]
        sampling_fn: Callable FloatTensor of size [*, n_codes] -> LongTensor of size [*]
        return: LongTensor of size [batch, 1, n_codebooks]

        The batch_size here is supposed to be 1.
        Uses embedding to predict codes autoregressively.
        Makes n_codes steps of autoregression.
        Each step includes applying of forward method, taking the last logits sequence and sampling from it.
        """
        B, d = embedding.shape
        device = embedding.device
        assert B == 1, "Batch size should be 1"

        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


        return codes_sequence


class EncoderDecoder(nn.Module):
    def __init__(self, d_model, n_phonemes, n_codes, n_codebooks):
        super().__init__()

        self.phoneme_embedding = nn.Embedding(num_embeddings=n_phonemes, embedding_dim=d_model)

        assert d_model % n_codebooks == 0, f"{d_model=} {n_codebooks=}"
        self.codes_embedding = nn.ModuleList([
            nn.Embedding(num_embeddings=n_codes, embedding_dim=d_model // n_codebooks)
            for _ in range(n_codebooks)
        ])

        self.phones_positional_encoding = nn.Embedding(num_embeddings=1000, embedding_dim=d_model)
        self.n_pos_embs = 2300
        self.codes_positional_encoding = nn.Embedding(num_embeddings=self.n_pos_embs, embedding_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=8,
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=8,
        )

    def forward(
        self,
        phones, # [B, l]
        phones_mask, # [B, l]
        codes, # [B, L, N]
        codes_mask, # [B, L]
        speaker_embs, # [B, d]
    ):
        """
        phones: LongTensor of size [batch, phones_len]
        phones_mask: BoolTensor of size [batch, phones_len]
        codes: LongTensor of size [batch, codes_len, n_codebooks]
        codes_mask: BoolTensor of size [batch, codes_len]
        speaker_embs: FloatTensor of size [batch, d_model]

        Method uses phonemes and speaker_embedding to create phoneme_embeddings.
        Then applies tranformer decoder in teacher-forcing regime.
        This method returns embeddings for further usage in SubDecoder.
        """
        device=phones.device

        # Embedding phonemes and concatenating with speaker embedding
        phone_embs = self.phoneme_embedding(phones)
        phone_embs = torch.cat((speaker_embs.unsqueeze(dim=1), phone_embs), dim=1)
        mask_complement = torch.ones(phones.shape[0], 1, device=device, dtype=torch.bool)
        phones_mask = torch.cat((mask_complement, phones_mask), dim=1)

        # Simple position encoding for phonemes
        phones_range = torch.arange(phones.shape[1] + 1, device=device).unsqueeze(dim=0)
        phones_PE = self.phones_positional_encoding(phones_range)
        phones_inp = phone_embs + phones_PE

        # Creating embeddings for each input codec, and concatenating them
        codes_embs = [emb_layer(codes[:, :, idx]) for idx, emb_layer in enumerate(self.codes_embedding)]
        codes_embs = torch.cat(codes_embs, dim=2)

        # Simple position encoding for codecs sequence
        codes_range = torch.arange(codes.shape[1], device=device).unsqueeze(dim=0)
        codes_range = torch.clamp(codes_range, 0, self.n_pos_embs - 1) # Needed to avoid out of range errors
        codes_PE = self.codes_positional_encoding(codes_range)
        codes_inp = codes_embs + codes_PE

        # Applying encoder, which calculates representations of phonemes
        phonemes_encoded = self.encoder(
            src=phones_inp,
            mask=None,
            src_key_padding_mask=~phones_mask
        )
        # Applying decoder, which calculates representations, which will be further used for subdecoder to predict codecs
        embeddings = self.decoder(
            tgt=codes_inp,
            memory=phonemes_encoded,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(codes.shape[1], device=device).bool(),
            memory_mask=None,
            tgt_key_padding_mask=~codes_mask,
            memory_key_padding_mask=~phones_mask,
        )

        return embeddings


class TTSTransformer(nn.Module):
    def __init__(self, n_phonemes: int, n_codes: int, n_codebooks: int):
        super().__init__()
        d_model = 512 + 256

        self.speaker_linear = nn.Linear(512, d_model)

        self.encoder_decoder = EncoderDecoder(
            d_model=d_model,
            n_phonemes=n_phonemes,
            n_codes=n_codes,
            n_codebooks=n_codebooks,
        )

        self.subdecoder = SubDecoder(
            d_model=d_model,
            n_codes=n_codes,
            n_codebooks=n_codebooks,
        )

    def forward(
        self,
        phones, # [B, l]
        phones_mask, # [B, l]
        codes, # [B, L, N]
        codes_mask, # [B, L]
        speaker_embs, # [B, ]
    ):
        """
        phones: LongTensor of size [batch, phones_len]
        phones_mask: BoolTensor of size [batch, phones_len]
        codes: LongTensor of size [batch, codes_len, n_codebooks]
        codes_mask: BoolTensor of size [batch, codes_len]
        speaker_embs: FloatTensor of size [batch, d_model]
        return: logits FloatTensor of size [batch, codes_len, n_codes, n_codes_in_codebooks]
        
        Method applies bioemb_linear, encoder, decoder and subdecoder.
        Works in teacher-forcing regime.
        """
        # Bring speaker_embedding to the neededd size
        speaker_embs = self.speaker_linear(speaker_embs)

        # Applying encoder and decoder, getting the codes representations for predictions of the next codes
        embeddings = self.encoder_decoder(
            phones=phones,
            phones_mask=phones_mask,
            codes=codes,
            codes_mask=codes_mask,
            speaker_embs=speaker_embs,
        )

        # [:, :, :-1, :] is needed, because the third dimension is of size n_codes + 1, because we concatenated embeding in subdecoder
        prediction = self.subdecoder(embeddings, codes)[:, :, :-1, :]

        return prediction

    @torch.no_grad()
    def autoregressive_sampling(
        self,
        phones, # [B, l]
        speaker_embs, # [B]
        max_size: int = 1000,
        start_token: int = 161,
        end_token: int = 160,
        sampling_fn: Callable = lambda x: x.argmax(dim=-1),
    ):
        """
        phones: LongTensor of size [batch, phones_len]
        phones_mask: BoolTensor of size [batch, phones_len]
        speaker_embs: FloatTensor of size [batch, d_model]
        max_size: int - the maximum number of steps in autoregressiong. Needed to avoid infinite inference if the end token does not generate.
        start_token: int - the start token index. The first codec vector in input codes sequence should be initialized with this value.
        end_token: int - the end token index. If the model predicts this token ,
        sampling_fn: Callable FloatTensor of size [*, n_codes] -> LongTensor of size [*]
        
        The batch_size here is supposed to be 1.
        Implements the same idea as forward, but instead of ground-truth codecs uses codecs, predicted and sampled autoregressively.
        Makes either max_size steps of autoregression or stops when end_token is predicted.
        On each step of autoregression calls TTSTransformer.forward and SubDecoder.autoregressive_sampling.
        As batch_size here allways equal to 1, the masks of phonemes and codecs are allways filled with ones here.
        """
        batch_size = phones.shape[0]
        assert batch_size == 1, "Batch size must be 1"
        device = phones.device

        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


        return codes
