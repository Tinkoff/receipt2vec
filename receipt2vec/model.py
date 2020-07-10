import os
import csv
import torch
import sentencepiece as spm
from allennlp.common.params import Params

from typing import List
from .utils import Utils
from tqdm import tqdm
from .models import BaseEncoder


class Receipt2vecEncoder():
    def __init__(
        self,
        encoder_model: str = None,
        bpe_model: str = None,
        price_array: List[float] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize model
        :param encoder_model str: Name of encoder model class
        :param bpe_model str: Path to bpe model file
        :param price_array List[float]: List of price groups of minimal value
        :param device torch.device: default torch.device('cpu')
        """
        encoder_model = encoder_model or 'BiTransformerCnnEncoder'
        if bpe_model is None:
            bpe_model = os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                'data',
                'bpe.model'
            ))
        if price_array is None:
            price_array = [
                0.0, 20.37, 30.9, 40.0,
                49.0, 58.0, 69.0, 81.55,
                98.0, 115.5, 139.15, 168.55,
                215.0, 296.75, 440.0, 854.0,
            ]

        params = Params({'type': encoder_model})
        encoder = BaseEncoder.from_params(params=params)
        self.encoder = encoder.load_from_last_checkpoint(device)
        self.bpe = self._load_bpe(bpe_model)
        self._utils = Utils(price_array=price_array)
        self._device = device

    def _load_bpe(self, model: str) -> spm.SentencePieceProcessor:
        print('Load BPE model')
        if model == '':
            print('Error: not set bpe model')
        try:
            bpe = spm.SentencePieceProcessor(model)
        except:  # noqa E722
            print('Error: cant load bpe model')
            exit(1)
        return bpe

    def __call__(self, receipt: str, price: float) -> torch.Tensor:
        """
        Transform receipt data to vec
        :param receipt: str - sring of receipt
        :param price: float - price of receipt
        :return: torch.Tensor - vextor of receipt
        """
        receipt_bpe_array = self._utils.prepare_receipt(receipt, self.bpe, self.encoder.max_bpe_token)
        price_group = self._utils.get_price_group(price)
        receipt_tensor = torch.cat([
            torch.tensor(receipt_bpe_array[:300]).long(),
            torch.zeros((max(300 - len(receipt_bpe_array), 0))).long()
        ], dim=0).long()
        price_tensor = torch.Tensor([price_group]).long()
        vec = self.encoder(torch.unsqueeze(receipt_tensor, 0), price_tensor)
        return torch.squeeze(vec, 0)

    def transform_file(
        self,
        receipts_file: str,
        out_file: str,
        write_header: bool = False,
        use_columns: List[str] = None,
        batch_size=128
    ):
        """
        Transform receipts file to vectors
        :param receipts_file: [str] Path to the file with receipts
        :param out_file: [str] Path to the result file
        :param write_header: [bool] If true, the output file contains column names
        :param use_columns: [List[str]] The name of the columns from input file
        that are written to the output file
        """
        data_loader = self._utils.data_loader(
            receipts_file, self.bpe, self.encoder.max_bpe_token, batch_size=batch_size, use_columns=use_columns
        )
        data_length = self._utils.get_data_length(receipts_file)
        total = data_length // batch_size if data_length // batch_size == 0 else data_length // batch_size + 1
        fieldnames = ['vector'] if use_columns is None else use_columns + ['vector']
        with open(out_file, 'w') as ofile:
            writer = csv.DictWriter(ofile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            with tqdm(total=total) as t:
                for receipts, prices, meta in data_loader:
                    receipts, prices = receipts.long().to(self._device), prices.long().to(self._device)
                    vectors = self.encoder(receipts, prices).cpu()

                    for inx, vector in enumerate(vectors):
                        result = {} if meta is None else meta[inx]
                        result['vector'] = ','.join([str(x) for x in vector.detach().numpy()])
                        writer.writerow(result)
                    t.update(1)
        print('Done')
