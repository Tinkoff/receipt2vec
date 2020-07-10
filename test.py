from unittest import TestCase
from tempfile import TemporaryDirectory
from receipt2vec.model import Receipt2vecEncoder
from receipt2vec.models.BaseEncoder import BaseEncoder

import torch
import csv
import os
import subprocess


@BaseEncoder.register("TestModelCls")
class TestModelClass(BaseEncoder):
    path_to_model = ''
    max_bpe_token = 10

    def forward(self, receipt, price):
        return torch.zeros((receipt.shape[0], 5))

    @classmethod
    def load_from_last_checkpoint(cls, device):
        encoder = cls().to(device)
        return encoder.eval()


class TestModel(TestCase):
    def setUp(self):
        self.encoder = Receipt2vecEncoder(encoder_model='TestModelCls', device=torch.device('cpu'))

    def tearDown(self):
        pass

    @staticmethod
    def create_intput_file(tempdir: str, set_test_column: bool = False) -> str:
        test_in_file = os.path.join(tempdir, 'in.csv')
        with open(test_in_file, 'w') as in_file:
            fieldnames = ['test', 'receipt', 'price'] if set_test_column else ['receipt', 'price']
            writer = csv.DictWriter(in_file, fieldnames=fieldnames)
            writer.writeheader()
            for x in range(10):
                test_receipt_dict = {}
                if set_test_column:
                    test_receipt_dict['test'] = x
                test_receipt_dict['receipt'] = 'молоко домик в деревне 1 л'
                test_receipt_dict['price'] = 49.99
                writer.writerow(test_receipt_dict)
        return test_in_file

    def test_predict_one(self):
        test_receipt = 'молоко домик в деревне 1 л'
        test_price = 49.99
        vec = self.encoder(test_receipt, test_price)
        self.assertIsInstance(vec, torch.Tensor)
        self.assertIsNotNone(vec)
        self.assertEqual(float(vec.max()), 0)

    def test_prefict_from_file(self):
        with TemporaryDirectory() as tempdir:
            test_in_file = self.create_intput_file(tempdir)

            test_out_file = os.path.join(tempdir, 'result.csv')
            self.encoder.transform_file(
                test_in_file,
                test_out_file,
                batch_size=3,
            )
            self.assertEqual(os.path.exists(test_out_file), True)
            with open(test_out_file) as o_file:
                line = o_file.readline().strip()
                self.assertEqual(line.strip(), '"0.0,0.0,0.0,0.0,0.0"')

    def test_prefict_from_file_with_header(self):
        with TemporaryDirectory() as tempdir:
            test_in_file = self.create_intput_file(tempdir, set_test_column=True)

            test_out_file = os.path.join(tempdir, 'result.csv')
            self.encoder.transform_file(
                test_in_file,
                test_out_file,
                batch_size=3,
                write_header=True,
                use_columns=['test'],
            )
            self.assertEqual(os.path.exists(test_out_file), True)
            with open(test_out_file) as o_file:
                line = o_file.readline().strip()
                self.assertEqual(line, 'test,vector')
                line = o_file.readline().strip()
                self.assertEqual(line, '0,"0.0,0.0,0.0,0.0,0.0"')

    def test_cli_no_header(self):
        with TemporaryDirectory() as tempdir:
            test_in_file = self.create_intput_file(tempdir)
            test_out_file = os.path.join(tempdir, 'result.csv')

            out = subprocess.check_output(
                f'python -m receipt2vec -i {test_in_file} -o {test_out_file} --batch=3',
                shell=True
            )
            print(out)
            self.assertEqual(os.path.exists(test_out_file), True)

    def test_cli_with_header(self):
        with TemporaryDirectory() as tempdir:
            test_in_file = self.create_intput_file(tempdir, set_test_column=True)
            test_out_file = os.path.join(tempdir, 'result.csv')

            out = subprocess.check_output(
                f'python -m receipt2vec -i {test_in_file} -o {test_out_file} --batch=3 ' +  # noqa
                '--write_header=1 --use_columns="test"',
                shell=True
            )
            print(out)
            self.assertEqual(os.path.exists(test_out_file), True)
            with open(test_out_file) as o_file:
                line = o_file.readline().strip()
                self.assertEqual(line, 'test,vector')
