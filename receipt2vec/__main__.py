import argparse
import torch
from .model import Receipt2vecEncoder


def _init():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Path to input file with receipts and prices', required=True)
    parser.add_argument('-o', '--output', type=str, help='Path to output file with results', required=True)
    parser.add_argument('--batch', type=int, default=128, help="Batch size [Default 128]")
    parser.add_argument('--gpu', type=int, default=-1, help="Num of gpu. If cpu use -1 [Default -1]")
    parser.add_argument(
        '--bpe', type=str, default=None,
        help="Path to bpe model file. If None - used default model [Default None]",
    )
    parser.add_argument(
        '--encoder', type=str, default=None,
        help="Name of encoder model. If None - used default model [Default None]",
    )
    parser.add_argument(
        '--write_header', type=int, default=0,
        help='Write header to the output file [0 or 1. Default 0]'
    )
    parser.add_argument(
        '--use_columns', type=str, default=None,
        help="A string of columns separated by ',' from the input file that will be written to the output file"
    )
    args = parser.parse_args()
    if args.write_header not in [0, 1]:
        print('Error: Incorrect params "write_header". Must be 0 or 1')
        exit(1)
    if args.use_columns is not None:
        try:
            args.use_columns = args.use_columns.split(',') if ',' in args.use_columns else [args.use_columns]
        except Exception as e:
            print(f'Error: {e}')
            exit(1)
    return args


def main():
    args = _init()
    config = {}
    config['device'] = torch.device('cpu') if args.gpu == -1 else torch.device(f'cuda:{args.gpu}')
    if args.bpe is not None:
        config['bpe_model'] = args.bpe
    if args.encoder is not None:
        config['encoder_model'] = args.encoder
    model = Receipt2vecEncoder(**config)
    model.transform_file(
        args.input,
        args.output,
        batch_size=args.batch,
        write_header=bool(args.write_header),
        use_columns=args.use_columns,
    )


if __name__ == "__main__":
    main()
