import torch
import torch.nn as nn
import warnings
import os

from typing import T, Type
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer \
    import BidirectionalLanguageModelTransformer
from allennlp.modules.seq2vec_encoders.cnn_encoder \
    import CnnEncoder
from .BaseEncoder import BaseEncoder
warnings.filterwarnings('ignore')


@BaseEncoder.register('BiTransformerCnnEncoder')
class BiTransformerCnnEncoder(BaseEncoder):
    path_to_model = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'models_data',
        'BiTransformerCnnEncodermodels.tar'
    )
    max_bpe_token = 31932

    def __init__(self):
        super(BiTransformerCnnEncoder, self).__init__()
        self.item_embedding = nn.Embedding(31932, 256)
        self.price_embedding = nn.Embedding(16, 16)
        self.transformer_encoder = BidirectionalLanguageModelTransformer(
            input_dim=256,
            hidden_dim=2048,
            num_layers=2,
        )
        self.conv = CnnEncoder(
            embedding_dim=256 * 2,
            num_filters=128,
            output_dim=256,
        )
        self.out = nn.Linear(272, 256)

    def forward(self, item, price):
        item_embeddings = self.item_embedding(item)
        price_embeddings = self.price_embedding(price)

        padding_mask = item != 0
        transformer_encoder_out = self.transformer_encoder(item_embeddings, padding_mask)

        padding_mask = transformer_encoder_out.sum(dim=-1) != 0
        conv = self.conv(transformer_encoder_out, padding_mask)

        flatten = torch.cat([conv, price_embeddings], dim=-1)
        out = self.out(flatten)
        return out

    @classmethod
    def load_from_last_checkpoint(cls, device: torch.device) -> Type[T]:
        print('Load model')
        encoder = cls()
        checkpoint = torch.load(encoder.path_to_model, map_location=device)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        return encoder.eval()
