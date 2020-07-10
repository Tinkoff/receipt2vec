# Receipt2vec
Framework по переводу строки товара из чека в векторное представление. Это способ формирования признаков (feature enginering) на которых пользователи смогут строить свои модели - модели оттока, рекомендации и тп. В частности, обмениваться признаками с партнёрами.

# Установка
```bash
pip install receipt2vec
```
# Использование
## CLI 
Перевод тестового файла в формате CSV.
```bash
$ receipt2vec --help
usage: receipt2vec [-h] -i INPUT -o OUTPUT [--batch BATCH] [--gpu GPU]
                   [--bpe BPE] [--encoder ENCODER]
                   [--write_header WRITE_HEADER] [--use_columns USE_COLUMNS]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input file with receipts and prices
  -o OUTPUT, --output OUTPUT
                        Path to output file with results
  --batch BATCH         Batch size [Default 128]
  --gpu GPU             Num of gpu. If cpu use -1 [Default -1]
  --bpe BPE             Path to bpe model file. If None - used default model
                        [Default None]
  --encoder ENCODER     Name of encoder model. If None - used default model
                        [Default None]
  --write_header WRITE_HEADER
                        Write header to the output file [0 or 1. Default 0]
  --use_columns USE_COLUMNS
                        A string of columns separated by ',' from the input
                        file that will be written to the output file

```
### Пример входного файла
Файл должен сожержать 2 колонки с заголовками - `receipt[srting],price[float]`
```bash
$ head items_.csv 
receipt,price
"Бутылка 1,0 Литр",8.0
Борщ с фасолью и сметаной,46.0
БЗМЖ СЫР PRETTO МОЦАРЕЛЛА ДЛЯ ,109.9
"БАЛТИКА №3 Пиво свет фильтр паст 4,8",52.99
Аккумулятор холода  800 млLTAK0048,139.9
```
Использование 
```bash
$ receipt2vec -i items_.csv -o items.vec
```

## Импорт модели
```python
>>> from receipt2vec.model import Receipt2vecEncoder
>>> model = Receipt2vecEncoder()
>>> vec = model('БАЛТИКА №3 Пиво свет фильтр паст 4,8', 52.99)
>>> print(vec.shape)
torch.Size([256])
```