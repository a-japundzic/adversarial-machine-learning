# adversarial-machine-learning
Different adversarial attacks performed on VGG-11 Convolutional Neural Network

Performed variuous adversarial attacks and recorded the results. Each iteration was a combination of FGSM attack or defense and PGD attack or defense. 


## a) Standard Train, Standard Test

I reused the model from assignment 3 question 2a with 5 EPOCHs. This is the initial base line with the unaltered MINST dataset:

Graph:

![image.png](attachment:91bf5547-0bc8-446f-acce-2fd920572380:image.png)

Epoch 5:

```python
Epoch 5:
  Train Loss: 0.0069 || Train Accuracy: 0.9980
  Test Loss: 0.0292  || Test Accuracy: 0.9915
```

Original images from the MINST dataset (this is for comparison with the adversarial examples): 

![image.png](attachment:1d1fa8e9-313e-47bf-b3d9-12a7c0f8bad1:image.png)

## b) Standard Train, FGSM Test

### $\epsilon=0.2$

Graph:

![image.png](attachment:61bdccd9-77de-44ea-be77-3c2080a8ab94:image.png)

Here we notice that the test accuracy plummets from above 98 percent to well below 20 percent. Clearly the attacker is quite effective. 

Epoch 5:

```python
Epoch 5:
  Train Loss: 0.0074 || Train Accuracy: 0.9978
  Test Loss fgsm: 6.8185  || Test Accuracy fgsm: 0.1474
```

Adversarial Images:

![image.png](attachment:3a202bf3-f0d7-4d7f-88fd-7b816d8ddd61:image.png)

We notice that the images still appear like numbers (anyone familiar with numbers would easily identify them correctly) but they have a lot more noise. 

We can also see in the plot how the model mislabels the numbers. Some of the mislabels appear quite reasonable, like the 9 can be a poorly drawn 4 and the 2 is close to being an 8, while others like the 5 being labeled a 9 are much less reasonable.  

### $\epsilon=0.1$

Graph:

![image.png](attachment:8dcabc00-3987-4bdd-84e5-c9298bf30fba:image.png)

With epsilon set to 0.1, the test accuracy falls much less drastically, with EPOCH 3 producing a result of 80 percent accuracy. This attack is clearly much less effective.

Epoch 5: 

```python
Epoch 5:
  Train Loss: 0.0073 || Train Accuracy: 0.9979
  Test Loss FGSM: 1.3753  || Test Accuracy FGSM: 0.6938
```

Adversarial Images

![image.png](attachment:b2ad5b55-ca5d-47ca-b964-0747052e8f5d:image.png)

Here we see that there is much less noise added to the images and that the model is able to predict the labels much more accurately as a result. 

### $\epsilon=0.5$

Graph:

![image.png](attachment:19e50e3f-65ab-42cf-9b74-b3cc589de9f6:image.png)

With epsilon, we see the test accuracy drop to an impressive low of less than 1 percent. 

Epoch:

```python
Epoch 5:
  Train Loss: 0.0093 || Train Accuracy: 0.9972
  Test Loss fgsm: 8.3902  || Test Accuracy fgsm: 0.0973
```

Adversarial Images:

![image.png](attachment:080c829a-a63e-4cfc-a18f-f610e946b990:image.png)

Looking at the example, we see why the test accuracy was so low. There so much noise that the numbers are almost unrecognizable. Even to a human these would be hard to identify without the label above. 

## c) FGSM Train, FGSM Test

### $\epsilon=0.2$

Graph: 

![image.png](attachment:2b84637d-f164-437b-b1ac-7c422d6c6089:image.png)

With the adversarial training, we see that test accuracy jump backup to above 95%. Clearly, the effects of the attacker are much less prevalent when doing adversarial training.

Epoch 5:

```python
Epoch 5:
  Train Loss fgsm: 0.0393 || Train Accuracy fgsm: 0.9876
  Test Loss fgsm: 0.1368  || Test Accuracy fgsm: 0.9560
```

Adversarial Images: 

![image.png](attachment:0a40bdcc-ec26-4050-9f19-21bdb15cb601:image.png)

Here we see that the images have the same amount of noise, but the model is able to generate accurate predictions. 

## $\epsilon =0.1$

Graph:

![image.png](attachment:445d750f-b129-43d8-867c-bb3e65144c99:image.png)

Here we see that our already impressive accuracy score without adversarial training has gone up to above 98%. This is still an impressive increase, showing the effectiveness of this type of adversarial training on this type of attack. 

Epoch 5:

```python
Epoch 5:
  Train Loss fgsm: 0.0251 || Train Accuracy fgsm: 0.9924
  Test Loss fgsm: 0.0405  || Test Accuracy fgsm: 0.9881
```

Adversarial Images:

![image.png](attachment:14f5dc24-7995-4084-9aba-fc6339260121:image.png)

Here we see that the images are almost identical to the original ones (very little noise) and that the model is able to accurately predict all of them.

## $\epsilon=0.5$

Graph:

![image.png](attachment:fbe67226-9c14-4729-b6df-2e8a3ebdd69e:image.png)

Here we see by the last epoch that there was also a large improvement in accuracy. 

Epoch 5:

```python
Epoch 5:
  Train Loss fgsm: 0.0410 || Train Accuracy fgsm: 0.9870
  Test Loss fgsm: 4.1323  || Test Accuracy fgsm: 0.1562
```

Adversarial Images:

![image.png](attachment:81093032-ea7f-407b-8e9f-a548bfa2976d:image.png)

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

![image.png](attachment:2c79677c-1b3d-4b47-89c4-6cb275c1264f:image.png)

Here we see that the test accuracy still improved by a lot, showing the effectiveness of PGD on other attacks. 

Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.0990 || Train Accuracy pgd: 0.9676
  Test Loss fgsm: 0.3664  || Test Accuracy fgsm: 0.8804
```

## $\epsilon =0.1$, PGD train, FGSM test

Graph:

![image.png](attachment:827622cd-2935-4c3f-9b10-40b2a98f93e0:image.png)

We see the same thing here, with a large improvement over no adversarial training but less than training with FGSM

Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.1019 || Train Accuracy pgd: 0.9667
  Test Loss fgsm: 0.1004  || Test Accuracy fgsm: 0.9666
```

## $\epsilon =0.5$, PGD train, FGSM test

Graph:

![image.png](attachment:ab3b0a1c-8c0a-4c92-9a38-8c505c7aa697:image.png)

Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.0997 || Train Accuracy pgd: 0.9678
  Test Loss fgsm: 3.4948  || Test Accuracy fgsm: 0.1014
```

## $\epsilon =0.2$,  PGD train, PGD test

Graph:

![image.png](attachment:6c9fc36d-745f-4513-9de4-1423ed49d8a2:image.png)

Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.1018 || Train Accuracy pgd: 0.9665
  Test Loss pgd: 0.1156  || Test Accuracy pgd: 0.9629
```

Adversarial Image (PGD):

![image.png](attachment:a4f1d213-f109-482f-a27f-5664bbfcf83a:image.png)

## $\epsilon =0.1$,  PGD train, PGD test

Graph:

![image.png](attachment:5f46ec70-02dc-45d2-af8c-32934c2f69d5:image.png)

Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.0991 || Train Accuracy pgd: 0.9673
  Test Loss pgd: 0.1202  || Test Accuracy pgd: 0.9608
```

Adversarial Image (PGD):

![image.png](attachment:20ba33bc-3903-475d-99e0-05bd4b119af8:image.png)

## $\epsilon =0.5$,  PGD train, PGD test

Graph:

![image.png](attachment:319e2f39-864e-4aef-9197-3d0b9f77af43:image.png)

Epoch 5:

```python
Epoch 5:
  Train Loss pgd: 0.1034 || Train Accuracy pgd: 0.9655
  Test Loss pgd: 0.1269  || Test Accuracy pgd: 0.9598
```

Adversarial Image (PGD):

![image.png](attachment:abe96443-3331-4393-825c-7c6a59820f99:image.png)

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
