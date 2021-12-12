# tugas2
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0a391dc2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import datasets\n",
    "from sklearn.cluster import KMeans\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8e709fa1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5.1, 3.5, 1.4, 0.2],\n",
       "       [4.9, 3. , 1.4, 0.2],\n",
       "       [4.7, 3.2, 1.3, 0.2],\n",
       "       [4.6, 3.1, 1.5, 0.2],\n",
       "       [5. , 3.6, 1.4, 0.2],\n",
       "       [5.4, 3.9, 1.7, 0.4],\n",
       "       [4.6, 3.4, 1.4, 0.3],\n",
       "       [5. , 3.4, 1.5, 0.2],\n",
       "       [4.4, 2.9, 1.4, 0.2],\n",
       "       [4.9, 3.1, 1.5, 0.1],\n",
       "       [5.4, 3.7, 1.5, 0.2],\n",
       "       [4.8, 3.4, 1.6, 0.2],\n",
       "       [4.8, 3. , 1.4, 0.1],\n",
       "       [4.3, 3. , 1.1, 0.1],\n",
       "       [5.8, 4. , 1.2, 0.2],\n",
       "       [5.7, 4.4, 1.5, 0.4],\n",
       "       [5.4, 3.9, 1.3, 0.4],\n",
       "       [5.1, 3.5, 1.4, 0.3],\n",
       "       [5.7, 3.8, 1.7, 0.3],\n",
       "       [5.1, 3.8, 1.5, 0.3],\n",
       "       [5.4, 3.4, 1.7, 0.2],\n",
       "       [5.1, 3.7, 1.5, 0.4],\n",
       "       [4.6, 3.6, 1. , 0.2],\n",
       "       [5.1, 3.3, 1.7, 0.5],\n",
       "       [4.8, 3.4, 1.9, 0.2],\n",
       "       [5. , 3. , 1.6, 0.2],\n",
       "       [5. , 3.4, 1.6, 0.4],\n",
       "       [5.2, 3.5, 1.5, 0.2],\n",
       "       [5.2, 3.4, 1.4, 0.2],\n",
       "       [4.7, 3.2, 1.6, 0.2],\n",
       "       [4.8, 3.1, 1.6, 0.2],\n",
       "       [5.4, 3.4, 1.5, 0.4],\n",
       "       [5.2, 4.1, 1.5, 0.1],\n",
       "       [5.5, 4.2, 1.4, 0.2],\n",
       "       [4.9, 3.1, 1.5, 0.2],\n",
       "       [5. , 3.2, 1.2, 0.2],\n",
       "       [5.5, 3.5, 1.3, 0.2],\n",
       "       [4.9, 3.6, 1.4, 0.1],\n",
       "       [4.4, 3. , 1.3, 0.2],\n",
       "       [5.1, 3.4, 1.5, 0.2],\n",
       "       [5. , 3.5, 1.3, 0.3],\n",
       "       [4.5, 2.3, 1.3, 0.3],\n",
       "       [4.4, 3.2, 1.3, 0.2],\n",
       "       [5. , 3.5, 1.6, 0.6],\n",
       "       [5.1, 3.8, 1.9, 0.4],\n",
       "       [4.8, 3. , 1.4, 0.3],\n",
       "       [5.1, 3.8, 1.6, 0.2],\n",
       "       [4.6, 3.2, 1.4, 0.2],\n",
       "       [5.3, 3.7, 1.5, 0.2],\n",
       "       [5. , 3.3, 1.4, 0.2],\n",
       "       [7. , 3.2, 4.7, 1.4],\n",
       "       [6.4, 3.2, 4.5, 1.5],\n",
       "       [6.9, 3.1, 4.9, 1.5],\n",
       "       [5.5, 2.3, 4. , 1.3],\n",
       "       [6.5, 2.8, 4.6, 1.5],\n",
       "       [5.7, 2.8, 4.5, 1.3],\n",
       "       [6.3, 3.3, 4.7, 1.6],\n",
       "       [4.9, 2.4, 3.3, 1. ],\n",
       "       [6.6, 2.9, 4.6, 1.3],\n",
       "       [5.2, 2.7, 3.9, 1.4],\n",
       "       [5. , 2. , 3.5, 1. ],\n",
       "       [5.9, 3. , 4.2, 1.5],\n",
       "       [6. , 2.2, 4. , 1. ],\n",
       "       [6.1, 2.9, 4.7, 1.4],\n",
       "       [5.6, 2.9, 3.6, 1.3],\n",
       "       [6.7, 3.1, 4.4, 1.4],\n",
       "       [5.6, 3. , 4.5, 1.5],\n",
       "       [5.8, 2.7, 4.1, 1. ],\n",
       "       [6.2, 2.2, 4.5, 1.5],\n",
       "       [5.6, 2.5, 3.9, 1.1],\n",
       "       [5.9, 3.2, 4.8, 1.8],\n",
       "       [6.1, 2.8, 4. , 1.3],\n",
       "       [6.3, 2.5, 4.9, 1.5],\n",
       "       [6.1, 2.8, 4.7, 1.2],\n",
       "       [6.4, 2.9, 4.3, 1.3],\n",
       "       [6.6, 3. , 4.4, 1.4],\n",
       "       [6.8, 2.8, 4.8, 1.4],\n",
       "       [6.7, 3. , 5. , 1.7],\n",
       "       [6. , 2.9, 4.5, 1.5],\n",
       "       [5.7, 2.6, 3.5, 1. ],\n",
       "       [5.5, 2.4, 3.8, 1.1],\n",
       "       [5.5, 2.4, 3.7, 1. ],\n",
       "       [5.8, 2.7, 3.9, 1.2],\n",
       "       [6. , 2.7, 5.1, 1.6],\n",
       "       [5.4, 3. , 4.5, 1.5],\n",
       "       [6. , 3.4, 4.5, 1.6],\n",
       "       [6.7, 3.1, 4.7, 1.5],\n",
       "       [6.3, 2.3, 4.4, 1.3],\n",
       "       [5.6, 3. , 4.1, 1.3],\n",
       "       [5.5, 2.5, 4. , 1.3],\n",
       "       [5.5, 2.6, 4.4, 1.2],\n",
       "       [6.1, 3. , 4.6, 1.4],\n",
       "       [5.8, 2.6, 4. , 1.2],\n",
       "       [5. , 2.3, 3.3, 1. ],\n",
       "       [5.6, 2.7, 4.2, 1.3],\n",
       "       [5.7, 3. , 4.2, 1.2],\n",
       "       [5.7, 2.9, 4.2, 1.3],\n",
       "       [6.2, 2.9, 4.3, 1.3],\n",
       "       [5.1, 2.5, 3. , 1.1],\n",
       "       [5.7, 2.8, 4.1, 1.3],\n",
       "       [6.3, 3.3, 6. , 2.5],\n",
       "       [5.8, 2.7, 5.1, 1.9],\n",
       "       [7.1, 3. , 5.9, 2.1],\n",
       "       [6.3, 2.9, 5.6, 1.8],\n",
       "       [6.5, 3. , 5.8, 2.2],\n",
       "       [7.6, 3. , 6.6, 2.1],\n",
       "       [4.9, 2.5, 4.5, 1.7],\n",
       "       [7.3, 2.9, 6.3, 1.8],\n",
       "       [6.7, 2.5, 5.8, 1.8],\n",
       "       [7.2, 3.6, 6.1, 2.5],\n",
       "       [6.5, 3.2, 5.1, 2. ],\n",
       "       [6.4, 2.7, 5.3, 1.9],\n",
       "       [6.8, 3. , 5.5, 2.1],\n",
       "       [5.7, 2.5, 5. , 2. ],\n",
       "       [5.8, 2.8, 5.1, 2.4],\n",
       "       [6.4, 3.2, 5.3, 2.3],\n",
       "       [6.5, 3. , 5.5, 1.8],\n",
       "       [7.7, 3.8, 6.7, 2.2],\n",
       "       [7.7, 2.6, 6.9, 2.3],\n",
       "       [6. , 2.2, 5. , 1.5],\n",
       "       [6.9, 3.2, 5.7, 2.3],\n",
       "       [5.6, 2.8, 4.9, 2. ],\n",
       "       [7.7, 2.8, 6.7, 2. ],\n",
       "       [6.3, 2.7, 4.9, 1.8],\n",
       "       [6.7, 3.3, 5.7, 2.1],\n",
       "       [7.2, 3.2, 6. , 1.8],\n",
       "       [6.2, 2.8, 4.8, 1.8],\n",
       "       [6.1, 3. , 4.9, 1.8],\n",
       "       [6.4, 2.8, 5.6, 2.1],\n",
       "       [7.2, 3. , 5.8, 1.6],\n",
       "       [7.4, 2.8, 6.1, 1.9],\n",
       "       [7.9, 3.8, 6.4, 2. ],\n",
       "       [6.4, 2.8, 5.6, 2.2],\n",
       "       [6.3, 2.8, 5.1, 1.5],\n",
       "       [6.1, 2.6, 5.6, 1.4],\n",
       "       [7.7, 3. , 6.1, 2.3],\n",
       "       [6.3, 3.4, 5.6, 2.4],\n",
       "       [6.4, 3.1, 5.5, 1.8],\n",
       "       [6. , 3. , 4.8, 1.8],\n",
       "       [6.9, 3.1, 5.4, 2.1],\n",
       "       [6.7, 3.1, 5.6, 2.4],\n",
       "       [6.9, 3.1, 5.1, 2.3],\n",
       "       [5.8, 2.7, 5.1, 1.9],\n",
       "       [6.8, 3.2, 5.9, 2.3],\n",
       "       [6.7, 3.3, 5.7, 2.5],\n",
       "       [6.7, 3. , 5.2, 2.3],\n",
       "       [6.3, 2.5, 5. , 1.9],\n",
       "       [6.5, 3. , 5.2, 2. ],\n",
       "       [6.2, 3.4, 5.4, 2.3],\n",
       "       [5.9, 3. , 5.1, 1.8]])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris = datasets.load_iris()\n",
    "iris.data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b329dc4a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(150, 4)\n"
     ]
    }
   ],
   "source": [
    "print(iris.data.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "026122f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['sepal length (cm)',\n",
       " 'sepal width (cm)',\n",
       " 'petal length (cm)',\n",
       " 'petal width (cm)']"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris.feature_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d0173286",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5a8454ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['setosa', 'versicolor', 'virginica'], dtype='<U10')"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris.target_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6fef9a27",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = pd.DataFrame(iris.data, columns = ['Sepal_Lenght', 'Sepal_Width', 'Petal_Lenght', 'Petal_Width'])\n",
    "y = pd.DataFrame(iris.target, columns = ['Target'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b9bff459",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sepal_Lenght</th>\n",
       "      <th>Sepal_Width</th>\n",
       "      <th>Petal_Lenght</th>\n",
       "      <th>Petal_Width</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Sepal_Lenght  Sepal_Width  Petal_Lenght  Petal_Width\n",
       "0             5.1          3.5           1.4          0.2\n",
       "1             4.9          3.0           1.4          0.2\n",
       "2             4.7          3.2           1.3          0.2\n",
       "3             4.6          3.1           1.5          0.2\n",
       "4             5.0          3.6           1.4          0.2\n",
       "..            ...          ...           ...          ...\n",
       "145           6.7          3.0           5.2          2.3\n",
       "146           6.3          2.5           5.0          1.9\n",
       "147           6.5          3.0           5.2          2.0\n",
       "148           6.2          3.4           5.4          2.3\n",
       "149           5.9          3.0           5.1          1.8\n",
       "\n",
       "[150 rows x 4 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "aef8afcc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 1 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Target\n",
       "0         0\n",
       "1         0\n",
       "2         0\n",
       "3         0\n",
       "4         0\n",
       "..      ...\n",
       "145       2\n",
       "146       2\n",
       "147       2\n",
       "148       2\n",
       "149       2\n",
       "\n",
       "[150 rows x 1 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3887cdf6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
      " 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 3 1 3 1 3 1 3 3 3 3 1 3 1 3 3 1 3 1 3 1 1\n",
      " 1 1 1 1 1 3 3 3 3 1 3 1 1 1 3 3 3 1 3 3 3 3 3 1 3 3 2 1 2 2 2 2 3 2 2 2 1\n",
      " 1 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 1 2 2 2 1 2 2 2 1 2 2 2 1 1\n",
      " 2 1]\n"
     ]
    }
   ],
   "source": [
    "model = KMeans(n_clusters = 4)\n",
    "model.fit(x)\n",
    "print(model.labels_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "fd7bdae4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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dZNOgAvuJcE8L881uTRr5gLVoEGuRtvQPPaYy6X3eFxnqsmTq57JUyIzs2bBHNyXYsuW8RmnPZj1FDUmgc2Z4yaUCM6ASU6OfDMvKQk2bHBBvsThh7sAHpya5/fDjBkuuDcE99wD//d/odfo97umj2uuiVux6oyNh25ME3XddeXTGaNlZhQWhk6lNneum6UnPM86N9cF0IagoABuuKH8Nebmws03ly97orOAbkwCq+6P4Jkzy57Pnh292aKh/LheudINeRDNsmX1V5aGwAK6MQmsurnQHTuWPW/fPvpkFvHIsY6kZcvIMwmBq703lHLWFwvoxiSwq68OzWCpzO23lz0fOTLy5Md+vztuQ9CsGYwZ41IGgyUnu8HI6qKzVUNmAd2YBHb22WWdZ7KyXBZHhw5wzDHlt735ZjfWdwmfDz76yNXaMzPd/ikprjPNuefW3zVUZupUlzuelubKmJYGBxwATz0V75LVP8tyMaYJWLcOvvrKjYd+4IEuvW/NGpfeVzJzfUZG5H2Li13q4+bNLv2xTZv6LXtVLV7sxp3p0QN69Yp3aeqOZbkYk4BufmAZ/lYb8CTn06z9Wh55fiUzZsA++7haaocO8Oijbts2bVzTxNChLpjn5rpZh95+G9580417Hu3mosfj9hszpuJg/t570LOnO3fnzvDMM27ih6uvdr1SjznGjT0eSUGBq2kfdJD70njggegZKp9+Cscd53qyXn45LF/ulu+9t1teUTD//Xf3S2TAADfm+0svVT9fvUGLNq5uXT9sPHRjau74839UKA48NOh5sYaP7X3BBaH75uSo7ruvG7c7ePzvUaNUi4trVp5JkyKPp56c7B7B57n55tB9CwtVDz00dFxyv191wADV/PzQbR9+OHQ7n081K0t14cLKy7hxo2rnzqHjl6enl39/GjoqGA/dAroxjcz2nAKFoipNUFHy2LKlbP/7748+qcMHH9SsTMFBu7JHaqrqunVl+06bppqREbk8jz9ett22bZHLDaojRlRexquvVk1JKb9vWprqt9/W7LrjoaKAbk0uxjQyDz2zisgjVkf39NNlz194IfLY6Tk58Oqr1S/Pzz9Xb8q1pCQ3OFiJl1+GHTsil+eFF8pef/559DTKmTMrbzp5+WXYubP88sJC1/SUCCygG9PING9W/QFDsrLKnocP91rC642+riLVSYuEsuFp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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "colormap = np.array(['Red', 'green', 'blue'])\n",
    "plt.scatter(x.Petal_Lenght, x.Petal_Width, c = colormap[iris.target], s = 40)\n",
    "plt.title('Before clustering with K-Means = Petal')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "60635e6f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "colormap = np.array(['Red', 'green', 'blue', 'pink'])\n",
    "plt.scatter(x.Petal_Lenght, x.Petal_Width, c = colormap[model.labels_], s = 40)\n",
    "plt.title('After clustering with K-Means - Petal')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "4e09724a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>col_0</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>row_0</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>50</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "      <td>27</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>32</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "col_0   0   1   2   3\n",
       "row_0                \n",
       "0      50   0   0   0\n",
       "1       0  23   0  27\n",
       "2       0  17  32   1"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.crosstab(iris.target, model.labels_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "24928cd7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f711d6ea",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[50  0  0  0]\n",
      " [ 0 23  0 27]\n",
      " [ 0 17 32  1]\n",
      " [ 0  0  0  0]]\n"
     ]
    }
   ],
   "source": [
    "result = confusion_matrix(iris.target, model.labels_)\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3dbbe538",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
