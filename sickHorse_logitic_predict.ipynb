{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logistic Regression for  Predict Hores"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Process"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 444,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib \n",
    "from matplotlib import pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 445,
   "metadata": {},
   "outputs": [],
   "source": [
    "def colicData():\n",
    "    frTrain=open('horseColicTraining.txt')\n",
    "    frTest=open('horseColicTest.txt')\n",
    "    Trainset=[]\n",
    "    Testset=[]\n",
    "    TrainLabels=[]\n",
    "    TestLabels=[]\n",
    "    for line in frTrain.readlines():\n",
    "        Trainline=line.strip().split(\"\\t\")\n",
    "        \n",
    "        lineArr=[]\n",
    "        \n",
    "        for i in range(21):\n",
    "            lineArr.append(float(Trainline[i]))\n",
    "                           \n",
    "        Trainset.append(lineArr)\n",
    "        \n",
    "        TrainLabels.append(float(Trainline[-1]))\n",
    "                           \n",
    "   \n",
    "    for lines in frTest.readlines():\n",
    "                           \n",
    "        Testline=lines.strip().split(\"\\t\")\n",
    "                           \n",
    "        TestlineArray=[]\n",
    "                           \n",
    "        for i in range(21):\n",
    "                           \n",
    "            TestlineArray.append(float(Testline[i]))\n",
    "                           \n",
    "        Testset.append(TestlineArray)\n",
    "                           \n",
    "        TestLabels.append(float(Testline[-1])) \n",
    "        \n",
    "    \n",
    "    Testset=np.array(Testset).reshape(-1,21)\n",
    "    \n",
    "    TestLabels=np.array(TestLabels).reshape(-1,1)\n",
    "    \n",
    "    Trainset=np.array(Trainset).reshape(-1,21)\n",
    "    \n",
    "    TrainLabels=np.array(TrainLabels).reshape(-1,1)\n",
    "    \n",
    "    return Testset,TestLabels,Trainset,TrainLabels\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 446,
   "metadata": {},
   "outputs": [],
   "source": [
    "def signmoid(Data):\n",
    "    \n",
    "    h=1.0/(1+np.exp(-Data))\n",
    "    \n",
    "    return h"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##  Training Process"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 447,
   "metadata": {},
   "outputs": [],
   "source": [
    "def StoGradientDescent(TrainSet,TrainLabels,nummerIter=1000):\n",
    "    m,n=TrainSet.shape\n",
    "    \n",
    "    Weights=np.ones(n).reshape(-1,1)\n",
    "    \n",
    "    for j in range(nummerIter):\n",
    "        \n",
    "        dataindex=range(m)\n",
    "        \n",
    "        dataindex=list(dataindex)\n",
    "        \n",
    "        Error=0\n",
    "        \n",
    "        for i in range(m):\n",
    "            \n",
    "            alpha=4/(1.0+j+i)+0.01\n",
    "            \n",
    "            randIndex=int(np.random.uniform(0,len(dataindex)))\n",
    "            \n",
    "            h=signmoid(np.dot(TrainSet[i],Weights)\n",
    "                      )\n",
    "                      \n",
    "            Error=TrainLabels[randIndex]-h\n",
    "            \n",
    "            U=(Error*TrainSet[randIndex]).reshape(21,1)\n",
    "                      \n",
    "            Weights=Weights+U\n",
    "                      \n",
    "            del(dataindex[randIndex])\n",
    "    return Weights"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " ## Test "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 448,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Test(Weights,Testset,TestLabels):\n",
    "    error=0\n",
    "    m,n=Testset.shape\n",
    "    Predict=signmoid(np.dot(Testset,Weights))\n",
    "    for i in range(m):\n",
    "        if Predict[i] == TestLabels[i]:\n",
    "            print(\"the %d th sample is right\" %i)\n",
    "        else:\n",
    "            print(\"the %d th sample is wrong\" %i)\n",
    "            error +=1\n",
    "    ErrorRate=float(error)/m\n",
    "    print(\"all %d sample the \" %m)\n",
    "    print(\"ErrorRate is %f\" %ErrorRate)\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data visualization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "######  incomplete #####"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Program Run"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 450,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "the 0 th sample is right\n",
      "the 1 th sample is right\n",
      "the 2 th sample is right\n",
      "the 3 th sample is wrong\n",
      "the 4 th sample is right\n",
      "the 5 th sample is right\n",
      "the 6 th sample is wrong\n",
      "the 7 th sample is right\n",
      "the 8 th sample is right\n",
      "the 9 th sample is right\n",
      "the 10 th sample is wrong\n",
      "the 11 th sample is right\n",
      "the 12 th sample is right\n",
      "the 13 th sample is wrong\n",
      "the 14 th sample is wrong\n",
      "the 15 th sample is wrong\n",
      "the 16 th sample is right\n",
      "the 17 th sample is right\n",
      "the 18 th sample is right\n",
      "the 19 th sample is wrong\n",
      "the 20 th sample is wrong\n",
      "the 21 th sample is right\n",
      "the 22 th sample is right\n",
      "the 23 th sample is right\n",
      "the 24 th sample is right\n",
      "the 25 th sample is right\n",
      "the 26 th sample is right\n",
      "the 27 th sample is wrong\n",
      "the 28 th sample is right\n",
      "the 29 th sample is wrong\n",
      "the 30 th sample is right\n",
      "the 31 th sample is wrong\n",
      "the 32 th sample is wrong\n",
      "the 33 th sample is right\n",
      "the 34 th sample is wrong\n",
      "the 35 th sample is right\n",
      "the 36 th sample is right\n",
      "the 37 th sample is right\n",
      "the 38 th sample is right\n",
      "the 39 th sample is right\n",
      "the 40 th sample is right\n",
      "the 41 th sample is wrong\n",
      "the 42 th sample is wrong\n",
      "the 43 th sample is right\n",
      "the 44 th sample is wrong\n",
      "the 45 th sample is wrong\n",
      "the 46 th sample is right\n",
      "the 47 th sample is right\n",
      "the 48 th sample is right\n",
      "the 49 th sample is right\n",
      "the 50 th sample is right\n",
      "the 51 th sample is wrong\n",
      "the 52 th sample is wrong\n",
      "the 53 th sample is wrong\n",
      "the 54 th sample is right\n",
      "the 55 th sample is right\n",
      "the 56 th sample is right\n",
      "the 57 th sample is wrong\n",
      "the 58 th sample is right\n",
      "the 59 th sample is wrong\n",
      "the 60 th sample is right\n",
      "the 61 th sample is right\n",
      "the 62 th sample is wrong\n",
      "the 63 th sample is right\n",
      "the 64 th sample is wrong\n",
      "the 65 th sample is right\n",
      "the 66 th sample is right\n",
      "all 67 sample the \n",
      "ErrorRate is 0.358209\n"
     ]
    }
   ],
   "source": [
    "Testset,Testlabels,Trainset,Trainlabes=colicData()\n",
    "Weights=StoGradientDescent(Trainset,Trainlabes,100)\n",
    "Test(Weights,Testset,Testlabels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
