# Per-class metrics: cnn1d_w100_s50_f8-16

## Three classes: leave-one-room-out

| Held-out room | Class | Precision | Recall | F1 | Windows |
| --- | --- | --- | --- | --- | --- |
| john | empty | 80.0% | 44.3% | 57.0% | 533 |
| john | still | 91.7% | 29.9% | 45.0% | 556 |
| john | moving | 44.5% | 97.1% | 61.1% | 519 |
| saifLiving | empty | 82.9% | 13.1% | 22.7% | 221 |
| saifLiving | still | 52.9% | 94.6% | 67.8% | 224 |
| saifLiving | moving | 96.1% | 100.0% | 98.0% | 222 |
| tim-living-room | empty | 66.7% | 35.5% | 46.4% | 557 |
| tim-living-room | still | 51.2% | 81.9% | 63.0% | 554 |
| tim-living-room | moving | 92.2% | 80.1% | 85.7% | 558 |

| Class | Precision (mean of rooms) | Recall (mean of rooms) | F1 (mean of rooms) |
| --- | --- | --- | --- |
| empty | 76.5% | 31.0% | 42.0% |
| still | 65.3% | 68.8% | 58.6% |
| moving | 77.6% | 92.4% | 81.6% |
| **Macro F1** | | | 60.7% |

## Three classes: all-rooms model, same-session test

| Class | Precision | Recall | F1 | Windows |
| --- | --- | --- | --- | --- |
| empty | 82.4% | 93.9% | 87.8% | 314 |
| still | 91.8% | 80.4% | 85.7% | 321 |
| moving | 97.4% | 96.2% | 96.8% | 313 |

## Motion view (quiet = empty + still): leave-one-room-out

| Held-out room | Class | Precision | Recall | F1 | Windows |
| --- | --- | --- | --- | --- | --- |
| john | moving | 44.5% | 97.1% | 61.1% | 519 |
| john | quiet | 96.8% | 42.3% | 58.9% | 1089 |
| saifLiving | moving | 96.1% | 100.0% | 98.0% | 222 |
| saifLiving | quiet | 100.0% | 98.0% | 99.0% | 445 |
| tim-living-room | moving | 92.2% | 80.1% | 85.7% | 558 |
| tim-living-room | quiet | 90.6% | 96.6% | 93.5% | 1111 |

| Class | Precision (mean of rooms) | Recall (mean of rooms) | F1 (mean of rooms) |
| --- | --- | --- | --- |
| moving | 77.6% | 92.4% | 81.6% |
| quiet | 95.8% | 79.0% | 83.8% |
| **Macro F1** | | | 82.7% |

## Motion view (quiet = empty + still): all-rooms model, same-session test

| Class | Precision | Recall | F1 | Windows |
| --- | --- | --- | --- | --- |
| moving | 97.4% | 96.2% | 96.8% | 313 |
| quiet | 98.1% | 98.7% | 98.4% | 635 |
