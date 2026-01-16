# adversarial-machine-learning
Different adversarial attacks performed on VGG-11 Convolutional Neural Network

Performed variuous adversarial attacks and recorded the results. Each iteration was a combination of FGSM attack or defense and PGD attack or defense. 


## a) Standard Train, Standard Test

I reused the model from assignment 3 question 2a with 5 EPOCHs. This is the initial base line with the unaltered MINST dataset:

Graph:

<img width="1188" height="886" alt="image" src="https://github.com/user-attachments/assets/f0f8a271-a661-45e4-8028-777c95348cdb" />


Epoch 5:

```python
Epoch 5:
  Train Loss: 0.0069 || Train Accuracy: 0.9980
  Test Loss: 0.0292  || Test Accuracy: 0.9915
```

Original images from the MINST dataset (this is for comparison with the adversarial examples): 

<img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/cad0ddd2-09c1-47ea-9e39-c64355712bb3" />


## b) Standard Train, FGSM Test

### $\epsilon=0.2$

Graph:

<img width="1192" height="887" alt="image" src="https://github.com/user-attachments/assets/47df7d7d-521c-4d69-bf74-310d14f0d7a8" />


Here we notice that the test accuracy plummets from above 98 percent to well below 20 percent. Clearly the attacker is quite effective. 

Epoch 5:

```python
Epoch 5:
  Train Loss: 0.0074 || Train Accuracy: 0.9978
  Test Loss fgsm: 6.8185  || Test Accuracy fgsm: 0.1474
```

Adversarial Images:

<img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/1382169f-b14b-448b-8baf-8b0134535a4d" />


We notice that the images still appear like numbers (anyone familiar with numbers would easily identify them correctly) but they have a lot more noise. 

We can also see in the plot how the model mislabels the numbers. Some of the mislabels appear quite reasonable, like the 9 can be a poorly drawn 4 and the 2 is close to being an 8, while others like the 5 being labeled a 9 are much less reasonable.  

### $\epsilon=0.1$

Graph:

<img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/99604b1b-10b9-426f-b101-3146b8386932" />


With epsilon set to 0.1, the test accuracy falls much less drastically, with EPOCH 3 producing a result of 80 percent accuracy. This attack is clearly much less effective.

Epoch 5: 

```python
Epoch 5:
  Train Loss: 0.0073 || Train Accuracy: 0.9979
  Test Loss FGSM: 1.3753  || Test Accuracy FGSM: 0.6938
```

Adversarial Images

<img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/8e4225ed-d471-45f4-8ffc-86b67d12bff7" />


Here we see that there is much less noise added to the images and that the model is able to predict the labels much more accurately as a result. 

### $\epsilon=0.5$

Graph:

<img width="1188" height="897" alt="image" src="https://github.com/user-attachments/assets/a472ac78-902c-45cf-96c6-42826a178673" />


With epsilon, we see the test accuracy drop to an impressive low of less than 1 percent. 

Epoch:

```python
Epoch 5:
  Train Loss: 0.0093 || Train Accuracy: 0.9972
  Test Loss fgsm: 8.3902  || Test Accuracy fgsm: 0.0973
```

Adversarial Images:

<img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/9f1a185e-ea0a-423a-91a3-acd322257508" />


Looking at the example, we see why the test accuracy was so low. There so much noise that the numbers are almost unrecognizable. Even to a human these would be hard to identify without the label above. 

## c) FGSM Train, FGSM Test

### $\epsilon=0.2$

Graph: 

<img width="1192" height="891" alt="image" src="https://github.com/user-attachments/assets/afa69643-e2fa-474b-8a5e-07e9ca4ae8bf" />


With the adversarial training, we see that test accuracy jump backup to above 95%. Clearly, the effects of the attacker are much less prevalent when doing adversarial training.

Epoch 5:

```python
Epoch 5:
  Train Loss fgsm: 0.0393 || Train Accuracy fgsm: 0.9876
  Test Loss fgsm: 0.1368  || Test Accuracy fgsm: 0.9560
```

Adversarial Images: 

<img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/dcae81c0-6a84-4bb8-8d3c-b6dcca20c460" />


Here we see that the images have the same amount of noise, but the model is able to generate accurate predictions. 

## $\epsilon =0.1$

Graph:

<img width="1199" height="897" alt="image" src="https://github.com/user-attachments/assets/e501a87d-46ad-460f-aea0-5a2479c3ce67" />


Here we see that our already impressive accuracy score without adversarial training has gone up to above 98%. This is still an impressive increase, showing the effectiveness of this type of adversarial training on this type of attack. 

Epoch 5:

```python
Epoch 5:
  Train Loss fgsm: 0.0251 || Train Accuracy fgsm: 0.9924
  Test Loss fgsm: 0.0405  || Test Accuracy fgsm: 0.9881
```

Adversarial Images:

<img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/1375a1ee-d345-46d8-9b03-27f738dcc9b5" />


Here we see that the images are almost identical to the original ones (very little noise) and that the model is able to accurately predict all of them.

## $\epsilon=0.5$

Graph:

<img width="1193" height="873" alt="image" src="https://github.com/user-attachments/assets/ee3efce8-d04a-4418-b3ac-0d4918112588" />


Here we see by the last epoch that there was also a large improvement in accuracy. 

Epoch 5:

```python
Epoch 5:
  Train Loss fgsm: 0.0410 || Train Accuracy fgsm: 0.9870
  Test Loss fgsm: 4.1323  || Test Accuracy fgsm: 0.1562
```

Adversarial Images:

<img width="1193" height="873" alt="image" src="https://github.com/user-attachments/assets/27a53a97-8ee4-4c67-8569-0913168d757b" />


Even though the images are hard to recognize even to humans, the machine learning model is able to predict with 15% accuracy. 

Overall, by performing adversarial training, we saw massive upticks in accuracy across all epsilon, showing the effectiveness of this type of training on this type of attack.

## d)

All PGD training was done with:

$$
\epsilon=0.2, \eta=0.02
$$

and $5$  iterations (to save time).

## $\epsilon =0.2$, PGD train, FGSM test

Graph:

<img width="1176" height="896" alt="image" src="https://github.com/user-attachments/assets/a7aa31d6-7c4e-4250-b05d-1d2f5b3f0203" />


Here we see that the test accuracy still improved by a lot, showing the effectiveness of PGD on other attacks. 

Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.0990 || Train Accuracy pgd: 0.9676
  Test Loss fgsm: 0.3664  || Test Accuracy fgsm: 0.8804
```

## $\epsilon =0.1$, PGD train, FGSM test

Graph:

<img width="1187" height="883" alt="image" src="https://github.com/user-attachments/assets/d36d6e17-5b5c-4108-98cc-12ee095f279d" />


We see the same thing here, with a large improvement over no adversarial training but less than training with FGSM

Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.1019 || Train Accuracy pgd: 0.9667
  Test Loss fgsm: 0.1004  || Test Accuracy fgsm: 0.9666
```

## $\epsilon =0.5$, PGD train, FGSM test

Graph:

<img width="1179" height="886" alt="image" src="https://github.com/user-attachments/assets/3aafe997-7b3e-4097-b0d5-b549b747965e" />


Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.0997 || Train Accuracy pgd: 0.9678
  Test Loss fgsm: 3.4948  || Test Accuracy fgsm: 0.1014
```

## $\epsilon =0.2$,  PGD train, PGD test

Graph:

<img width="1188" height="882" alt="image" src="https://github.com/user-attachments/assets/1140823b-d938-474e-a4a2-77c0e1836ea3" />


Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.1018 || Train Accuracy pgd: 0.9665
  Test Loss pgd: 0.1156  || Test Accuracy pgd: 0.9629
```

Adversarial Image (PGD):

<img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/00a690d8-f35a-45b6-b2dc-02673a422912" />


## $\epsilon =0.1$,  PGD train, PGD test

Graph:

<img width="1182" height="894" alt="image" src="https://github.com/user-attachments/assets/6f596136-6ef9-4f0a-a5d1-e3f4ea8df26c" />


Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.0991 || Train Accuracy pgd: 0.9673
  Test Loss pgd: 0.1202  || Test Accuracy pgd: 0.9608
```

Adversarial Image (PGD):

<img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/72f34e97-7492-448b-b2e0-29f6aa92fbcc" />


## $\epsilon =0.5$,  PGD train, PGD test

Graph:

<img width="1197" height="884" alt="image" src="https://github.com/user-attachments/assets/fcd46ecb-04f0-4acf-8c0f-065c2da22b62" />


Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.1034 || Train Accuracy pgd: 0.9655
  Test Loss pgd: 0.1269  || Test Accuracy pgd: 0.9598
```

Adversarial Image (PGD):

<img width="1000" height="200" alt="image" src="https://github.com/user-attachments/assets/4184ce45-2f35-42cd-9c31-de249969d1a8" />


For all three of the above tests, we see similar results to what we saw with FGSM train and FGSM test. 

## Tables (Test Accuracy):

All adversarial training was done with $\epsilon =0.2$ and (when relevant) $\eta=0.02$ and iterations $5$. 

The results are recorded after the fifth EPOCH. 

### Table, $\epsilon =0.1$:

|  | Standard Training | FGSM Adversarial Training | PGD Adversarial Training |
| --- | --- | --- | --- |
| Standard Test Accuracy | 0.9915 | 0.9842 | 0.9938 |
| Robust Test Accuracy (FGSM) | 0.6938 | 0.9881 | 0.9666 |
| Robust Test Accuracy (PGD) | 0.4882 | 0.5447 | 0.9608 |

### Table, $\epsilon =0.2$:

|  | Standard Training | FGSM Adversarial Training | PGD Adversarial Training |
| --- | --- | --- | --- |
| Standard Test Accuracy | 0.9915 | 0.9842 | 0.9938 |
| Robust Test Accuracy (FGSM) | 0.1474 | 0.9560 | 0.8804 |
| Robust Test Accuracy (PGD) | 0.4946 | 0.6628 | 0.9629 |

### Table, $\epsilon =0.5$:

|  | Standard Training | FGSM Adversarial Training | PGD Adversarial Training |
| --- | --- | --- | --- |
| Standard Test Accuracy | 0.9915 | 0.9842 | 0.9938 |
| Robust Test Accuracy (FGSM) | 0.0973 | 0.1562 | 0.1014 |
| Robust Test Accuracy (PGD) | 0.5220 | 0.6418 | 0.9598 |

Some interesting trends is that doing adversarial training did not affect the standard test accuracy notably. With PGD adversarial training having less of an effect on standard test accuracy than FSGM adversarial training.

Also, changing epsilon had a massive effect on FGSM robust test accuracy but did not have a large impact on PGD. This could be because of the other hyperparameters I chose (eta and iterations).
