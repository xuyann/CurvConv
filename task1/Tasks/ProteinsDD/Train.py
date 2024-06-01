'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file Train.py

    \brief Code to train a classification network on the enzyme task.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import logging
import math
import time
import numpy as np
import configparser
import argparse

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.dirname(BASE_DIR)
ROOT_PROJ_DIR = os.path.dirname(TASKS_DIR)
sys.path.append(ROOT_PROJ_DIR)

from IEProtLib.tf_utils import Trainer
from IEProtLib.tf_utils import TrainLoop
from IEProtLib.models.mol import ProtClass
from IEProtLib.py_utils import visualize_progress
from Datasets import ProtClassProteinsDB

current_milli_time = lambda: time.time() * 1000.0

class ProtClassTrainLoop(TrainLoop):
    """Class to train a classification network on the protclass100 dataset.
    """
    
    def __init__(self, pConfigFile):
        """Constructor.

        Args:
            pConfigFile (string): Path to the configuration file.
        """

        #Load the configuration file.
        self.config_ = configparser.ConfigParser()
        self.config_.read(pConfigFile)

        #Load the parameters.
        trainConfigDict = self.config_._sections['ProteinsDB']
        self.batchSize_ = int(trainConfigDict['batchsize'])
        self.augment_ = trainConfigDict['augment'] == "True"
        self.maxModelsSaved_ = int(trainConfigDict['maxmodelssaved'])
        self.aminoInput_ = trainConfigDict['aminoinput'] == "True"
        self.foldId_ = trainConfigDict['foldid']

        #Call the constructor of the parent.
        TrainLoop.__init__(self, self.config_._sections['TrainLoop'])

        #Save the config file in the log folder.
        os.system('cp %s %s' % (pConfigFile, self.logFolder_))

        #Initialize best accuracy-
        self.bestAccuracy_ = 0.0
        self.bestClassAccuracy_ = 0.0


    def __create_datasets__(self):
        """Method to create the datasets.
        """

        print("")
        print("########## Loading training dataset")
        self.trainDS_ = ProtClassProteinsDB(
            pDataset = "Training", 
            pPath = "./Datasets/ProteinsDB/",
            pFoldId = self.foldId_,
            pAmino = self.aminoInput_,
            pLoadText = False)
        print(self.trainDS_.get_num_proteins(), "proteins loaded")

        print("")
        print("########## Loading test dataset")
        self.testDS_ = ProtClassProteinsDB(
            pDataset = "Validation", 
            pPath = "./Datasets/ProteinsDB/",
            pFoldId = self.foldId_,
            pAmino = self.aminoInput_,
            pLoadText = False)  
        print(self.testDS_.get_num_proteins(), "proteins loaded")


    def __create_model__(self):
        """Method to create the model.
        """

        #Create the model object.
        self.model_ = ProtClass(self.config_._sections['ProtClass'], 
                3, self.batchSize_, 1, self.aminoInput_)
        
        #Create the placeholders.
        if not self.aminoInput_:
            self.numInFeatures_ = self.model_.create_placeholders(4)
        else:
            self.numInFeatures_ = self.model_.create_placeholders(0)

        #Create the model.
        self.model_.create_model(self.epochStep_, self.numEpochs_)

        #Create the loss function.        
        self.loss_ = self.model_.create_loss()
        self.probs_ = tf.nn.sigmoid(self.model_.logits_)

        #Create accuracy
        predictions = tf.cast(tf.greater(self.probs_, 0.5), tf.int32)
        labels = tf.cast(self.model_.labelsPH_, tf.int32)
        correct = tf.equal(predictions, labels)
        self.accuracy_ = (tf.reduce_sum(input_tensor=tf.cast(correct, tf.float32)) / 
            float(self.batchSize_))*100.0


    def __create_trainers__(self):
        """Method to create the trainer objects.
        """
        self.trainer_ = Trainer(self.config_._sections['Trainer'], 
            self.epochStep_, self.loss_, pCheckNans=True) 


    def __create_savers__(self):
        """Method to create the saver objects.
        """
        self.saver_ = tf.compat.v1.train.Saver(max_to_keep=self.maxModelsSaved_)


    def __create_tf_summaries__(self):
        """Method to create the tensorflow summaries.
        """

        self.accuracyPH_ = tf.compat.v1.placeholder(tf.float32)
        self.lossPH_ = tf.compat.v1.placeholder(tf.float32)

        #Train summaries.
        lossSummary = tf.compat.v1.summary.scalar('Loss', self.lossPH_)
        accuracySummary = tf.compat.v1.summary.scalar('Accuracy', self.accuracyPH_)
        lrSummary = tf.compat.v1.summary.scalar('LR', self.trainer_.learningRate_)
        self.trainingSummary_ = tf.compat.v1.summary.merge([lossSummary, 
            accuracySummary, lrSummary])
        
        #Test summaries.
        testLossSummary = tf.compat.v1.summary.scalar('Test_Loss', self.lossPH_)
        testAccuracySummary = tf.compat.v1.summary.scalar('Test_Accuracy', self.accuracyPH_)
        self.testSummary_ = tf.compat.v1.summary.merge([testLossSummary, testAccuracySummary])

    def __compute_curvature__(self):
        
        #self.trainDS_.compute_curvature(self.augment_)
        #self.testDS_.compute_curvature(self.augment_)
        pass
    def __train_one_epoch__(self, pNumEpoch):
        """Private method to train one epoch.

        Args:
            pNumEpoch (int): Current number of epoch.
        """

        #Calculate num batches.
        numBatchesTrain = self.trainDS_.get_num_proteins()//self.batchSize_

        #Init dataset epoch.
        self.trainDS_.start_epoch()

        #Process each batch.
        accumAccuracy = 0.0
        accumLoss = 0.0
        accumCounter = 1.0
        
        for curBatch in range(numBatchesTrain):

            #Get the starting time.
            startDataProcess = current_milli_time()
            
            #Get the batch data.
            
            protBatch, features, labels, _, _ = self.trainDS_.get_next_batch(True,
                self.batchSize_, self.augment_)
            
            #Create the dictionary for tensorflow.
            curDict = {}
            
            self.model_.associate_inputs_to_ph(curDict, protBatch, features, 
                np.array(labels).reshape((-1, 1)), True)
            
            #Get the end time of the pre-process.
            endDataProcess = current_milli_time()
            
            #Execute a training step.
            curAccuracy, curLoss, wl2Loss, _ = self.sess_.run(
                [self.accuracy_, self.loss_, 
                self.trainer_.weightLoss_,
                self.trainer_.trainOps_], curDict)
            
            #Get the end time of the computation.
            endComputation = current_milli_time()

            #Accumulate the accuracy and loss.
            accumAccuracy += (curAccuracy - accumAccuracy)/accumCounter
            accumLoss += (curLoss - accumLoss)/accumCounter
            accumCounter += 1.0

            #Visualize process.
            if curBatch% 10 == 0 and curBatch > 0:
                visualize_progress(curBatch, numBatchesTrain, 
                    "Loss: %.6f (%.1f) | Accuracy: %.4f | (Data: %.2f ms | TF: %.2f ms) " %
                    (accumLoss, wl2Loss, accumAccuracy,  
                    endDataProcess-startDataProcess, 
                    endComputation-endDataProcess),
                    pSameLine = True)
        print()

        #Write the sumary.
        trainSumm = self.sess_.run(self.trainingSummary_, 
            {self.accuracyPH_ : accumAccuracy, 
            self.lossPH_ : accumLoss})
        self.summaryWriter_.add_summary(trainSumm, pNumEpoch)

    
    def __test_one_epoch__(self, pNumEpoch):
        """Private method to test one epoch.

        Args:
            pNumEpoch (int): Current number of epoch.
        """

        #Calculate num batches.
        numBatchesTest = self.testDS_.get_num_proteins()//self.batchSize_
        if self.testDS_.get_num_proteins()%self.batchSize_ != 0:
            numBatchesTest += 1

        #Init dataset epoch.
        self.testDS_.start_epoch()

        #Test the model.
        accumCounter = 1.0
        accumTestLoss = 0.0
        accumTestAccuracy = 0.0
        for curBatch in range(numBatchesTest):

            #Get the batch data.
            protBatch, features, labels, names, validProts = self.testDS_.get_next_batch(False,self.batchSize_)

            #Create the dictionary for tensorflow.
            curDict = {}
            self.model_.associate_inputs_to_ph(curDict, protBatch, features, 
                np.array(labels).reshape((-1,1)), False)

            #Execute a training step.
            curProbs, curLoss = self.sess_.run(
                [self.probs_, self.loss_], curDict)

            #Accum acuracy.
            for curModel in range(len(labels)):
                curPrediction = 0
                if curProbs[curModel] > 0.5:
                    curPrediction = 1
                
                curAccuracy = 0.0
                if curPrediction == labels[curModel]:
                    curAccuracy = 100.0
                accumTestAccuracy +=  (curAccuracy - accumTestAccuracy)/accumCounter 
                accumCounter += 1
            accumTestLoss += curLoss

            if curBatch% 10 == 0 and curBatch > 0:
                visualize_progress(curBatch, numBatchesTest, pSameLine = True)

        #Print the result of the test.
        totalLoss = accumTestLoss/float(numBatchesTest) 
        print("End test:")
        print("Accuracy: %.4f [%.4f]" % (accumTestAccuracy, self.bestAccuracy_))
        print("Loss: %.6f" % (totalLoss))

        #Write the summary.
        testSumm = self.sess_.run(self.testSummary_, 
            {self.accuracyPH_ : accumTestAccuracy, 
            self.lossPH_ : totalLoss})
        self.summaryWriter_.add_summary(testSumm, pNumEpoch)

        #Save the model.
        self.saver_.save(self.sess_, self.logFolder_+"/model.ckpt")

        if accumTestAccuracy > self.bestAccuracy_:
            self.bestAccuracy_ = accumTestAccuracy
            self.saver_.save(self.sess_, 
                self.logFolder_+"/best.ckpt", 
                global_step=self.epochStep_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train classification of proteins (ProteinsDB)')
    parser.add_argument('--configFile', default='train.ini', help='Configuration file (default: train.ini)')
    args = parser.parse_args()

    trainObj = ProtClassTrainLoop(args.configFile)
    #trainObj.get_curvature()
    trainObj.train()
