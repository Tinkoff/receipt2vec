import torch
import csv

from sentencepiece import SentencePieceProcessor
from typing import List, Tuple


class Utils():
    def __init__(self, price_array: List[float]):
        self.price_array = price_array

    def get_price_group(self, price: float) -> int:
        for price_group in range(len(self.price_array)):
            if price < self.price_array[price_group]:
                return price_group - 1
        return len(self.price_array) - 1

    @staticmethod
    def prepare_receipt(receipt: str, bpe: SentencePieceProcessor, max_token: int = 0) -> List[int]:
        max_token = max_token if max_token > 0 else bpe.piece_size()
        receipt_bpe_array = [token if token < max_token else 0 for token in bpe.encode(receipt)]
        return receipt_bpe_array

    @staticmethod
    def get_data_length(in_file: str) -> int:
        length = 0
        with open(in_file) as file:
            for i, line in enumerate(file):
                if i == 0 or line.strip() == '':
                    continue
                length += 1
                if length % 1_000 == 0:
                    _l = str(length // 1000) + 'K' if length < 1_000_000 else str(length // 1_000_000) + 'M'
                    print(f'Get file len: {_l}', end='\r')
        print(f'Total file length: {length}')
        return length

    def data_loader(
        self,
        in_file: str,
        bpe: SentencePieceProcessor,
        max_token: int = 0,
        batch_size: int = 128,
        use_columns: List[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        receipts = torch.zeros((batch_size, 300)).long()
        prices = torch.zeros(batch_size).long()
        metas = list()
        with open(in_file, 'r') as file:
            reader = csv.DictReader(file)
            columns = ['receipt', 'price']
            columns = columns if use_columns is None else columns + use_columns
            for column in columns:
                if column not in reader.fieldnames:
                    print(f'Error: Column "{column}" not found in input file')
                    exit(1)
            inx = 0
            for line in reader:
                if line['receipt'].strip() == '':
                    continue
                receipt = self.prepare_receipt(line['receipt'], bpe, max_token)
                price_group = self.get_price_group(float(line['price']))
                receipt_tensor = torch.cat([
                    torch.tensor(receipt[:300]).long(),
                    torch.zeros((max(300 - len(receipt), 0))).long()
                ], dim=0).long()
                meta = None \
                    if use_columns is None \
                    else {key: value for key, value in line.items() if key in use_columns}
                receipts[inx % batch_size] += receipt_tensor
                prices[inx % batch_size] += price_group
                metas.append(meta)
                inx += 1
                if inx == batch_size:
                    yield receipts, prices, metas if use_columns is not None else None
                    inx = 0
                    receipts = torch.zeros((batch_size, 300)).long()
                    prices = torch.zeros(batch_size).long()
                    metas = list()
        if inx != 0:
            yield receipts[:inx], prices[:inx], metas if use_columns is not None else None
