# HRNIM-MRFA

### Requirements
~~~python
pip3 install -r requirements.txt
~~~

The model is trained on Composition-1K train dataset, and tested on Transparent-460 and UHR-395 test dataset. Model weight link [Google Drive](https://drive.google.com/file/d/1AFZSkaKjkVDM7Sn9B-fZiju3YqWKgJkO/view?usp=sharing)

### Transparent-460
| Models | SAD | MSE | Grad | Conn |
|  ----  | ----  |  ----  | ----  |  ----  |
| MEMatte-S | 151.60 | 10.22 | 28.82 | 141.86 |
| HRNIM-MRFA | 143.03 | 8.92 | 27.99 | 133.59 |

### UHR-395
| Models | SAD | MSE | Grad | Conn |
|  ----  | ----  |  ----  | ----  |  ----  |
| MEMatte-S | 623.93 | 11.13 | 220.09 | 572.94 |
| HRNIM-MRFA | 612.75 | 9.65 | 196.82 | 406.96 |


## Testing on Transparent-460 or UHR-395
Download the model file 'checkpoint/' and place it in the root directory.

1.Run the inference code
~~~python
python3 inference.py
~~~

2.Evaluate the results by the official evaluation python code evaluation.py
~~~python
python3 evaluation.py
~~~

Obtain the dataset from [Transparent-460](https://github.com/AceCHQ/TransMatting) and [UHR-395](https://github.com/linyiheng123/MEMatte). 

## Acknowledgment
This repo borrows code from several repos, like [MEMatte](https://github.com/linyiheng123/MEMatte)
