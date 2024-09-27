| Row | Uncertainty Method                | Calibration Method | Fridge MAE                 | Fridge ECE                    |
|-----|-----------------------------------|--------------------|----------------------------|-------------------------------|
| **Model: S2P (Homoscedastic)**          |                    |                            |                               |            |
| Before Calibration                      |                    |                            |                               |            |
| 1   | MC                                | -                  | 26.09 ± 1.40               | 0.211 ± 0.001                 |
| 2   | DE                                | -                  | 24.65 ± 0.50               | 0.268 ± 0.014                 |
| 3   | BS                                | -                  | 24.90 ± 0.58               | 0.182 ± 0.033                 |
| After Calibration                       |                    |                            |                               |            |
| 4   | MC                                | Isotonic           | 26.09 ± 1.40               | 0.210 ± 0.001                 |
| 5   | DE                                | Isotonic           | 24.65 ± 0.50               | 0.261 ± 0.015                 |
| 6   | BS                                | Isotonic           | 24.90 ± 0.58               | 0.215 ± 0.028                 |
| 7   | -                                 | Conformal          | 25.69 ± 1.16               | 0.192 ± 0.022                 |
| **Model: Gaussian S2P (Heteroscedastic)**|                   |                            |                               |            |
| Before Calibration                      |                    |                            |                               |            |
| 8   | Base S2P                          | -                  | 27.03 ± 0.46               | 0.130 ± 0.008                 |
| 9   | MC                                | -                  | 26.50 ± 0.54               | 0.049 ± 0.016                 |
| 10  | DE                                | -                  | 26.81 ± 0.42               | 0.077 ± 0.035                 |
| 11  | BS                                | -                  | 26.75 ± 0.28               | 0.117 ± 0.035                 |
| After Calibration                       |                    |                            |                               |            |
| 12  | Base S2P                          | Isotonic           | 27.03 ± 0.46               | 0.070 ± 0.015                 |
| 13  | MC                                | Isotonic           | 26.50 ± 0.54               | 0.191 ± 0.011                 |
| 14  | DE                                | Isotonic           | 26.81 ± 0.42               | 0.047 ± 0.012                 |
| 15  | BS                                | Isotonic           | 26.75 ± 0.28               | 0.078 ± 0.028                 |
| 16  | Base S2P                          | Conformal          | 27.03 ± 0.46               | 0.105 ± 0.013                 |
| **Model: S2P (Conformalized Quantile Regression)** |         |                            |                               |            |
| Before Calibration                      |                    |                            |                               |            |
| 17  | Base S2P                          | -                  | -                          | 0.078 ± 0.020                 |
| After Calibration                       |                    |                            |                               |            
| 18  | Base S2P                          | Conformal          | -                          | 0.078 ± 0.012                 |


| Row | Uncertainty Method                | Calibration Method | Dishwasher MAE             | Dishwasher ECE                |
|-----|-----------------------------------|--------------------|----------------------------|-------------------------------|
| **Model: S2P (Homoscedastic)**          |                    |                            |                               |            |
| Before Calibration                      |                    |                            |                               |            |
| 1   | MC                                | -                  | 10.96 ± 1.97               | 0.413 ± 0.029                 |
| 2   | DE                                | -                  | 13.45 ± 1.67               | 0.270 ± 0.140                 |
| 3   | BS                                | -                  | 11.38 ± 0.72               | 0.161 ± 0.134                 |
| After Calibration                       |                    |                            |                               |            |
| 4   | MC                                | Isotonic           | 10.96 ± 1.97               | 0.351 ± 0.055                 |
| 5   | DE                                | Isotonic           | 13.45 ± 1.67               | 0.252 ± 0.139                 |
| 6   | BS                                | Isotonic           | 11.38 ± 0.72               | 0.210 ± 0.114                 |
| 7   | -                                 | Conformal          | 10.88 ± 2.00               | 0.207 ± 0.036                 |
| **Model: Gaussian S2P (Heteroscedastic)**|                   |                            |                               |            |
| Before Calibration                      |                    |                            |                               |            |
| 8   | Base S2P                          | -                  | 9.62 ± 0.33                | 0.146 ± 0.067                 |
| 9   | MC                                | -                  | 9.42 ± 0.21                | 0.136 ± 0.032                 |
| 10  | DE                                | -                  | 9.69 ± 0.12                | 0.160 ± 0.041                 |
| 11  | BS                                | -                  | 10.57 ± 0.72               | 0.357 ± 0.036                 |
| After Calibration                       |                    |                            |                               |            |
| 12  | Base S2P                          | Isotonic           | 9.62 ± 0.33                | 0.078 ± 0.030                 |
| 13  | MC                                | Isotonic           | 9.42 ± 0.21                | 0.070 ± 0.064                 |
| 14  | DE                                | Isotonic           | 9.69 ± 0.12                | 0.061 ± 0.013                 |
| 15  | BS                                | Isotonic           | 10.57 ± 0.72               | 0.075 ± 0.028                 |
| 16  | Base S2P                          | Conformal          | 9.62 ± 0.33                | 0.088 ± 0.052                 |
| **Model: S2P (Conformalized Quantile Regression)** |         |                            |                               |            |
| Before Calibration                      |                    |                            |                               |            |
| 17  | Base S2P                          | -                  | -                          | 0.059 ± 0.028                 |
| After Calibration                       |                    |                            |                               |            
| 18  | Base S2P                          | Conformal          | -                          | 0.053 ± 0.019                 |

