| Front end     | Working resolution | Precision | Peak memory (GiB) | Seconds | Keypoints or matches | Outcome                      |
|---------------|--------------------|-----------|-------------------|---------|----------------------|------------------------------|
| RootSIFT      | 2048               | single    | 0.30              | 0.21    | 6404                 | fits on the device           |
| RootSIFT      | native             | single    | 0.09              | 1.08    | 6179                 | fits on the device           |
| DoG + HardNet | 2048               | single    | 1.71              | 0.36    | 6404                 | fits on the device           |
| DoG + HardNet | native             | single    | 1.65              | 1.28    | 6179                 | fits on the device           |
| SuperPoint    | 2048               | single    | 1.44              | 0.10    | 8192                 | fits on the device           |
| SuperPoint    | native             | single    | 9.61              | 1.89    | 8192                 | satisfied from system memory |
| ALIKED        | 2048               | single    | 4.44              | 0.15    | 8192                 | fits on the device           |
| ALIKED        | native             | single    | 29.80             | 20.94   | 8192                 | satisfied from system memory |
| DISK          | 1024               | single    | 1.86              | 0.10    | 7754                 | fits on the device           |
| DISK          | 1024               | half      | 0.93              | 0.05    | 7751                 | fits on the device           |
| DISK          | 1600               | single    | 4.18              | 0.17    | 8192                 | fits on the device           |
| DISK          | 1600               | half      | 2.08              | 0.08    | 8189                 | fits on the device           |
| DISK          | 2048               | single    | 6.71              | 3.22    | 8192                 | fits on the device           |
| DISK          | 2048               | half      | 3.34              | 0.13    | 8188                 | fits on the device           |
| DISK          | 3200               | single    | 15.99             | 38.20   | 8192                 | satisfied from system memory |
| DISK          | 3200               | half      | 7.96              | 14.39   | 8168                 | fits on the device           |
| DISK          | native             | single    |                   |         |                      | allocation failed            |
| DISK          | native             | half      | 15.04             | 596.87  | 8145                 | satisfied from system memory |
| LoFTR         | 1024               | single    | 2.16              | 0.44    | 7846                 | fits on the device           |
| LoFTR         | 1600               | single    | 11.39             | 7.21    | 18030                | satisfied from system memory |
| LoFTR         | 2048               | single    | 29.61             | 40.94   | 28241                | satisfied from system memory |
