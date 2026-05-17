---
layout: post
title: Fondamentaux du calcul optimisé avec Python CUDA (cours NVIDIA)
date: 2025-08-21
description: Découverte de Numba, création de kernels CUDA avec Python
tags: cuda numba 
categories: efficiency computing programming
thumbnail: 
---

Cet articles reprends mes notes sur le [cours de calcul optimisé avec CUDA Python de NVIDIA](https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-AC-10+V1&unit=block-v1:DLI+S-AC-10+V1+type@vertical+block@dd940fadc01947879e8acd2fc12f5707).

## Introduction à CUDA avec Numba

*Qu'est ce que Numba ?* Il s'agit d'un **compilateur de fonctions**, **juste-à-temps**, **spécialisé par types**, pour accélérer le code Python **axé sur les chiffres** pour un CPU ou un GPU.
### Définitions
* **Compilateur de fonctions** : Numba compile des fonctions Python, pas des applications entières, ni des parties de fonctions. Numba ne remplace pas votre interpréteur Python, mais est simplement un autre module Python qui peut transformer une fonction en une fonction (généralement) plus rapide.
* **Spécialisation par type** : Numba accélère votre fonction en générant une implémentation spécialisée pour les types de données spécifiques que vous utilisez. Les fonctions Python sont conçues pour fonctionner sur des types de données génériques, ce qui les rend très flexibles, mais aussi très lentes. En pratique, vous n'appellerez une fonction qu'avec un petit nombre de types d'arguments, donc Numba générera une implémentation rapide pour chaque ensemble de types.
* **Juste-à-temps** : Numba traduit les fonctions lorsqu'elles sont appelées pour la première fois. Cela garantit que le compilateur sait quels types d'arguments vous allez utiliser. Cela permet également d'utiliser Numba de manière interactive dans un notebook Jupyter aussi facilement qu'une application traditionnelle.
* **Axé sur les données numériques** : actuellement, Numba se concentre sur les types de données numériques, tels que int, float et complex. La prise en charge du traitement des chaînes de caractères est très limitée, et de nombreux cas d'utilisation de chaînes de caractères ne fonctionneront pas bien sur le GPU. Pour obtenir les meilleurs résultats avec Numba, vous utiliserez probablement des tableaux NumPy.

### Comparaison des alternatives

* **CUDA C/C++** : la façon la plus commune , la plus performante et la plus flexible d'utiliser CUDA
* **pyCUDA** : option la plus performante pour CUDA Python, requiert de rédiger du code C dans le code Python, et en général, beaucoup de modifications de code. Expose l'entière API CUDA C/C++
* **Numba** : potentiellement moins performant que pyCUDA, mais toujours une accélération massive souvent avec peu de modification de code.

## Premiers pas : compiler pour le CPU

On utilise le décorateur `@jit` : 

```python
from numba import jit
import math

# This is the function decorator syntax and is equivalent to `hypot = jit(hypot)`.
# The Numba compiler is just a function you can call whenever you want!
@jit
def hypot(x, y):
    # Implementation from https://en.wikipedia.org/wiki/Hypot
    x = abs(x);
    y = abs(y);
    t = min(x, y);
    x = max(x, y);
    t = t / x;
    return x * math.sqrt(1+t*t)
```

Certaines fonctionnalités python, comme les dictionnaires ne sont pas prises en compte par Numba. Par défaut, `@jit` compilera quand même. Mais si on veut obtenir des erreurs pour cela, il faut utiliser l'option `nopython`.

```python
@jit(nopython=True)
def cannot_compile(x):
    return x['key']

cannot_compile(dict(key='value'))
```


On peut également utiliser l'alias `@njit` :


```python
from numba import njit

@njit
def cannot_compile(x):
    return x['key']

cannot_compile(dict(key='value'))
```

Le mode nopython est la meilleure pratique car cela amène à une meilleure performance.

## Introduction à Numba pour les GPU avec les fonctions universelles numpy (ufuncs)

Les GPU sont des puces conçues pour le parallélisme de données. Le débit maximum est atteint lorsque le GPU calcule les mêmes opérations sur plusieurs éléments différents en même temps.

Les fonctions universelles Numpy qui performent la même opération sur tous les éléments d'un *array* numpy, sont naturellement parallèles en termes de données, ils fit donc bien sur un GPU.

Numba a la capacité de créer des fonctions universelles compilées, il suffit d'implémenter une fonctions scalaire, appliquée sur tout les éléments d'une entrée et Numba se chargera du reste, simplement avec un décorateur `@vectorize`.

```python
from numba import vectorize

@vectorize
def add_ten(num):
    return num + 10 # This scalar operation will be performed on each element

nums = np.arange(10)
add_ten(nums) # pass the whole array into the ufunc, it performs the operation on each element
```

Ici, cette fonction s'éxecute sur un CPU. Pour le faire sur GPU, il faut préciser le type de la sortie et de l'entrée, sous la forme :  
`return_value_type(argument1_value_type, argument2_value_type, ...)`  
Ainsi, sur GPU la fonction précédente se présente sous la forme :

```python
@vectorize(['int64(int64, int64)'], target='cuda') # Type signature and target are required for the GPU
def add_ufunc(x, y):
    return x + y
```

Cependant avec une telle fonction, la version GPU peut être moins performante car : 

* **Nos entrées sont trop petites** : le GPU atteint ses performances grâce au parallélisme, en traitant des milliers de valeurs à la fois. Nos entrées de test ne contiennent respectivement que 4 et 16 entiers. Nous avons besoin d'un tableau beaucoup plus grand pour occuper le GPU.
* **Nos calculs sont trop simples** : envoyer un calcul au GPU implique une charge assez importante par rapport à l'appel d'une fonction sur le CPU. Si nos calculs ne comportent pas suffisamment d'opérations mathématiques (souvent appelées « intensité arithmétique »), le GPU passera la plupart de son temps à attendre que les données soient transférées.
* **Nous copions les données vers et depuis le GPU** : bien que dans certains cas, le coût de la copie des données vers et depuis le GPU puisse en valoir la peine pour une seule fonction, il est souvent préférable d'exécuter plusieurs opérations GPU à la suite. Dans ces cas, il est logique d'envoyer les données au GPU et de les y conserver jusqu'à ce que tout notre traitement soit terminé.
* **Nos types de données sont plus volumineux que nécessaire** : notre exemple utilise int64 alors que nous n'en avons probablement pas besoin. Le code scalaire utilisant des types de données 32 et 64 bits s'exécute à peu près à la même vitesse sur le CPU, et pour les types entiers, la différence n'est peut-être pas considérable, mais les types de données à virgule flottante 64 bits peuvent avoir un coût de performance important sur le GPU, selon le type de GPU. Les opérations arithmétiques de base sur les flottants 64 bits peuvent être de 2 à 24 fois plus lentes (architecture Pascal Tesla) que sur les flottants 32 bits (architecture Maxwell GeForce). Si vous utilisez des GPU plus modernes (Volta, Turing, Ampere), cela pourrait être beaucoup moins préoccupant. NumPy utilise par défaut des types de données 64 bits lors de la création de tableaux. Il est donc important de définir l'attribut dtype ou d'utiliser la méthode ndarray.astype() pour choisir des types 32 bits lorsque vous en avez besoin.

## CUDA Device Functions

Les fonctions universelles sont très pratiques pour les opérations elementaires (`element-wise`) mais il existe de nombreuses fonctions qui ne remplissent pas cette condition. Ainsi pour compiler d'autres fonctions, on utilise `numba.cuda.jit`, qui est aussi un décorateur. 
Dans l'exemple ci dessous, il ne requiert pas de signature de type, et on lui passe deux valeur scalaires, contrairement aux fonctions `@vectorize` auxquelles on donnait des array numpy.

```python
from numba import cuda

@cuda.jit(device=True)
def polar_to_cartesian(rho, theta):
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    return x, y

@vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
def polar_distance(rho1, theta1, rho2, theta2):
    x1, y1 = polar_to_cartesian(rho1, theta1) # We can use device functions inside our GPU ufuncs
    x2, y2 = polar_to_cartesian(rho2, theta2)
    
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
```

## Gérer la mémoire du GPU

Selon le [guide des bonnes pratiques de CUDA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) :  
> **Priorité élevée** : minimiser le transfert de données entre l'hôte et le périphérique, même si cela implique d'exécuter certains noyaux sur le périphérique qui ne présentent pas de gains de performances par rapport à leur exécution sur le processeur de l'hôte.

Dans cette optique, nous devons réfléchir à la manière d'empêcher ce transfert automatique des données vers l'hôte afin de pouvoir effectuer des opérations supplémentaires sur les données, en ne payant le prix de leur recopie vers l'hôte que lorsque nous sommes vraiment prêts.

Pour ce faire, il suffit de créer des tableaux de périphériques CUDA et de les transmettre à nos fonctions GPU. Les tableaux de périphériques ne seront pas automatiquement transférés vers l'hôte après le traitement et pourront être réutilisés à notre guise sur le périphérique avant d'être finalement renvoyés, en tout ou en partie, vers l'hôte, uniquement si nécessaire.

Le module `numba.cuda` comprend une fonction qui copie les données hôtes vers le GPU et renvoie un tableau de périphériques CUDA. Notez que lorsque nous essayons d'imprimer le contenu du tableau de périphériques, nous obtenons uniquement des informations sur le tableau, et non son contenu réel. En effet, les données se trouvent sur le périphérique et nous devons les transférer vers l'hôte afin d'imprimer leurs valeurs.

```python
from numba import cuda

@vectorize(['float32(float32, float32)'], target='cuda')
def add_ufunc(x, y):
    return x + y

x_device = cuda.to_device(x)
y_device = cuda.to_device(y)
%timeit add_ufunc(x_device, y_device)
```

Comme x_device et y_device sont déjà sur le périphérique, ce benchmark est beaucoup plus rapide.

Cependant, nous continuons d'allouer un tableau de périphériques pour la sortie de l'ufunc et de le recopier vers l'hôte, même si dans la cellule ci-dessus, nous n'assignons pas réellement le tableau à une variable. Pour éviter cela, nous pouvons créer le tableau de sortie avec la fonction numba.cuda.device_array() :

```python
out_device = cuda.device_array(shape=(n,), dtype=np.float32)
%timeit add_ufunc(x_device, y_device, out=out_device)
```

Cet appel à add_ufunc n'implique aucun transfert de données entre l'hôte et le périphérique et s'exécute donc plus rapidement. Si nous souhaitons ramener un tableau de périphériques dans la mémoire hôte, nous pouvons utiliser la méthode copy_to_host() :

```python
out_host = out_device.copy_to_host()
print(out_host[:10])
```

