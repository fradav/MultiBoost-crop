/*
 *
 *    MultiBoost - Multi-purpose boosting package
 *
 *    Copyright (C) 2010   AppStat group
 *                         Laboratoire de l'Accelerateur Lineaire
 *                         Universite Paris-Sud, 11, CNRS
 *
 *    This file is part of the MultiBoost library
 *
 *    This library is free software; you can redistribute it 
 *    and/or modify it under the terms of the GNU General Public
 *    License as published by the Free Software Foundation; either
 *    version 2.1 of the License, or (at your option) any later version.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
 *
 *    Contact: Balazs Kegl (balazs.kegl@gmail.com)
 *             Norman Casagrande (nova77@gmail.com)
 *             Robert Busa-Fekete (busarobi@gmail.com)
 *
 *    For more information and up-to-date version, please visit
 *        
 *                       http://www.multiboost.org/
 *
 */


#include <ctime> // for time
#include <cmath> // for exp
#include <fstream> // for ofstream of the step-by-step data
#include <limits>
#include <iomanip> // setprecision

#include "Utils/Utils.h" // for addAndCheckExtension
#include "Defaults.h" // for defaultLearner
#include "IO/OutputInfo.h"
#include "IO/InputData.h"
#include "IO/Serialization.h" // to save the found strong hypothesis

#include "WeakLearners/BaseLearner.h"
#include "StrongLearners/AdaBoostMHLearner.h"
#include "Classifiers/AdaBoostMHClassifier.h"
#include "VJCascadeLearner.h"

namespace MultiBoost {
	
	// -----------------------------------------------------------------------------------
	
	void VJCascadeLearner::getArgs(const nor_utils::Args& args)
	{
		if ( args.hasArgument("verbose") )
			args.getValue("verbose", 0, _verbose);
		
		// The file with the step-by-step information
		if ( args.hasArgument("outputinfo") )
			args.getValue("outputinfo", 0, _outputInfoFile);
		
		///////////////////////////////////////////////////
		// get the output strong hypothesis file name, if given
		if ( args.hasArgument("shypname") )
			args.getValue("shypname", 0, _shypFileName);
		else
			_shypFileName = string(SHYP_NAME);
		
		_shypFileName = nor_utils::addAndCheckExtension(_shypFileName, SHYP_EXTENSION);
		
		///////////////////////////////////////////////////
		// get the output strong hypothesis file name, if given
		if ( args.hasArgument("shypcomp") )
			args.getValue("shypcomp", 0, _isShypCompressed );
		else
			_isShypCompressed = false;
		
		// get the name of the learner
		_baseLearnerName = "HaarSingleStumpLearner";
		if ( args.hasArgument("learnertype") )
			args.getValue("learnertype", 0, _baseLearnerName);
		
		if ( args.hasArgument("firstStage") )
			args.getValue("firstStage", 0, _stageStartNumber);
		
		
		// -train <dataFile> <nInterations>
		if ( args.hasArgument("train") )
		{
			cout << "Validation file is needed in VJ cascade!!!" << endl;
		}
		// -traintest <trainingDataFile> <testDataFile> <nInterations>
		else if ( args.hasArgument("traintest") ) 
		{
			args.getValue("traintest", 0, _trainFileName);
			args.getValue("traintest", 1, _validFileName);
			args.getValue("traintest", 2, _numIterations);
		}		
		// -traintest <trainingDataFile> <validDataFile> <testDataFile> <nInterations>
		else if ( args.hasArgument("trainvalidtest") ) 
		{
			args.getValue("trainvalidtest", 0, _trainFileName);
			args.getValue("trainvalidtest", 1, _testFileName);
			args.getValue("trainvalidtest", 2, _validFileName);
			args.getValue("trainvalidtest", 3, _numIterations);
		}
		
	}
	
	// -----------------------------------------------------------------------------------
	
	void VJCascadeLearner::run(const nor_utils::Args& args)
	{
		// load the arguments
		this->getArgs(args);
		
		double Fi=1.0;
		double prevFi=1.0;
		double Do=1.0;
		_foundHypotheses.resize(0);
		
		// get the registered weak learner (type from name)
		BaseLearner* pWeakHypothesisSource = 
		BaseLearner::RegisteredLearners().getLearner(_baseLearnerName);
		// initialize learning options; normally it's done in the strong loop
		// also, here we do it for Product learners, so input data can be created
		pWeakHypothesisSource->initLearningOptions(args);
		
		BaseLearner* pConstantWeakHypothesisSource = 
		BaseLearner::RegisteredLearners().getLearner("ConstantLearner");
		
		// get the training input data, and load it
		
		InputData* pTrainingData = pWeakHypothesisSource->createInputData();
		pTrainingData->initOptions(args);
		pTrainingData->load(_trainFileName, IT_TRAIN, _verbose);
		
		InputData* pValidationData = pWeakHypothesisSource->createInputData();
		pValidationData->initOptions(args);
		pValidationData->load(_validFileName, IT_TRAIN, _verbose);				
		
		// get the testing input data, and load it
		InputData* pTestData = NULL;
		if ( !_testFileName.empty() )
		{
			pTestData = pWeakHypothesisSource->createInputData();
			pTestData->initOptions(args);
			pTestData->load(_testFileName, IT_TEST, _verbose);
		}						
		
		Serialization ss(_shypFileName, false );
		ss.writeHeader(_baseLearnerName); // this must go after resumeProcess has been called
		
		
		if (_verbose == 1)
			cout << "Learning in progress..." << endl;
		
		vector<bool> _activeTrainInstances(pTrainingData->getNumExamples());
		fill(_activeTrainInstances.begin(),_activeTrainInstances.end(), true );
		
		vector<bool> _activeValidationInstances(pValidationData->getNumExamples());
		fill(_activeValidationInstances.begin(),_activeValidationInstances.end(), true );
		///////////////////////////////////////////////////////////////////////
		// Starting the Cascad main loop
		///////////////////////////////////////////////////////////////////////		
		for(int stagei=0; stagei < _numIterations; ++stagei )
		{
			// filter data
			set< int > ind;
			for(int i=0; i<_activeTrainInstances.size(); ++i) ind.insert(i);
			pTrainingData->loadIndexSet( ind );
			
			ind.clear();
			for(int i=0; i<_activeValidationInstances.size(); ++i) ind.insert(i);
			pValidationData->loadIndexSet( ind );
			
			resetWeights(pTrainingData);
			vector<double> posteriors(0);
			
			Fi=prevFi;
			Fi *= _maxAcceptableFalsePositiveRate;			
			
			int t=0;
			_foundHypotheses.resize( _foundHypotheses.size()+1 );
			///////////////////////////////////////////////////////////////////////
			// Starting the AdaBoost main loop
			///////////////////////////////////////////////////////////////////////
			while (true)
			{
				if (_verbose > 1)
					cout << "------- STAGE " << stagei << " WORKING ON ITERATION " << (t+1) << " -------" << endl;
				
				BaseLearner* pWeakHypothesis = pWeakHypothesisSource->create();
				pWeakHypothesis->initLearningOptions(args);				
				
				pWeakHypothesis->setTrainingData(pTrainingData);
				
				float energy = pWeakHypothesis->run();
				
				//float gamma = pWeakHypothesis->getEdge();
				//cout << gamma << endl;
				
				if ( (_withConstantLearner) || ( energy != energy ) ) // check constant learner if user wants it (if energi is nan, then we chose constant learner
				{
					BaseLearner* pConstantWeakHypothesis = pConstantWeakHypothesisSource->create() ;
					pConstantWeakHypothesis->initLearningOptions(args);
					pConstantWeakHypothesis->setTrainingData(pTrainingData);
					float constantEnergy = pConstantWeakHypothesis->run();
					
					if ( (constantEnergy <= energy) || ( energy != energy ) ) {
						delete pWeakHypothesis;
						pWeakHypothesis = pConstantWeakHypothesis;
					}
				}
				
				if (_verbose > 1)
					cout << "Weak learner: " << pWeakHypothesis->getName()<< endl;
				// Output the step-by-step information
				//printOutputInfo(pOutInfo, t, pTrainingData, pTestData, pWeakHypothesis);
				
				// Updates the weights and returns the edge
				float gamma = updateWeights(pTrainingData, pWeakHypothesis);
				
				if (_verbose > 1)
				{
					cout << setprecision(5)
					<< "--> Alpha = " << pWeakHypothesis->getAlpha() << endl
					<< "--> Edge  = " << gamma << endl
					<< "--> Energy  = " << energy << endl
					//            << "--> ConstantEnergy  = " << constantEnergy << endl
					//            << "--> difference  = " << (energy - constantEnergy) << endl
					;
				}
				
				// If gamma <= theta the algorithm must stop.
				// If theta == 0 and gamma is 0, it means that the weak learner is no better than chance
				// and no further training is possible.
				if (gamma <= 0)
				{
					if (_verbose > 0)
					{
						cout << "Can't train any further: edge = " << gamma << endl;
					}
					
					//          delete pWeakHypothesis;
					//          break; 
				}
				
				// append the current weak learner to strong hypothesis file,
				// that is, serialize it.
				ss.appendHypothesis(t, pWeakHypothesis);
				
				// Add it to the internal list of weak hypotheses
				_foundHypotheses[stagei].push_back(pWeakHypothesis); 
				
				// evaluate current detector on validation set
				calculatePosteriors( pValidationData, _foundHypotheses[stagei], posteriors );
				
				
				
				
				t++;
				//delete pWeakHypothesis;
			}  // loop on iterations
			// filter data set
			
			
			
			
			
			//_maxAcceptableFalsePositiveRate
			//_minAcceptableDetectionRate
			
			/////////////////////////////////////////////////////////
			ss.appendStageSeparatorFooter();
			
		}// end of cascade
		
		
		// write the footer of the strong hypothesis file
		ss.writeFooter();
		
		// write the weights of the instances if the name of weights file isn't empty
		//printOutWeights( pTrainingData );
		
		
		// Free the two input data objects
		if (pTrainingData)
			delete pTrainingData;
		if (pValidationData)
			delete pValidationData;
		if (pTestData)
			delete pTestData;
		
		if (_verbose > 0)
			cout << "Learning completed." << endl;
	}
	
	// -------------------------------------------------------------------------
	void VJCascadeLearner::calculatePosteriors( InputData* pData, vector<BaseLearner*>& weakHypotheses, vector<double>& posteriors )
	{
		const int numExamples = pData->getNumExamples();		
		double sumAlpha=0.0;
		
		posteriors.resize(numExamples);
		fill( posteriors.begin(), posteriors.end(), 0.0 );

		vector<BaseLearner*>::iterator whyIt = weakHypotheses.begin();				
		for (;whyIt != weakHypotheses.end(); ++whyIt )
		{
			BaseLearner* currWeakHyp = *whyIt;
			float alpha = currWeakHyp->getAlpha();
			
			// for every point
			for (int i = 0; i < numExamples; ++i)
			{
				// just for the negative class
				posteriors[i] += alpha * currWeakHyp->classify(pData, i, 0);
			}			
		}
		/*
		for (int i = 0; i < numExamples; ++i)
		{
			// just for the negative class
			posteriors[i] /= sumAlpha;
		}			
		*/
	}
							 
											   
	// -------------------------------------------------------------------------							 
	void VJCascadeLearner::classify(const nor_utils::Args& args)
	{
		AdaBoostMHClassifier classifier(args, _verbose);
		
		// -test <dataFile> <shypFile>
		string testFileName = args.getValue<string>("test", 0);
		string shypFileName = args.getValue<string>("test", 1);
		int numIterations = args.getValue<int>("test", 2);
		
		string outResFileName;
		if ( args.getNumValues("test") > 3 )
			args.getValue("test", 3, outResFileName);
		
		classifier.run(testFileName, shypFileName, numIterations, outResFileName);
	}
	
	// -------------------------------------------------------------------------
	
	void VJCascadeLearner::doConfusionMatrix(const nor_utils::Args& args)
	{
		AdaBoostMHClassifier classifier(args, _verbose);
		
		// -cmatrix <dataFile> <shypFile>
		if ( args.hasArgument("cmatrix") )
		{
			string testFileName = args.getValue<string>("cmatrix", 0);
			string shypFileName = args.getValue<string>("cmatrix", 1);
			
			classifier.printConfusionMatrix(testFileName, shypFileName);
		}
		// -cmatrixfile <dataFile> <shypFile> <outFile>
		else if ( args.hasArgument("cmatrixfile") )
		{
			string testFileName = args.getValue<string>("cmatrix", 0);
			string shypFileName = args.getValue<string>("cmatrix", 1);
			string outResFileName = args.getValue<string>("cmatrix", 2);
			
			classifier.saveConfusionMatrix(testFileName, shypFileName, outResFileName);
		}
	}
	
	// -------------------------------------------------------------------------
	
	void VJCascadeLearner::doLikelihoods(const nor_utils::Args& args)
	{
		AdaBoostMHClassifier classifier(args, _verbose);
		
		// -posteriors <dataFile> <shypFile> <outFileName>
		string testFileName = args.getValue<string>("likelihood", 0);
		string shypFileName = args.getValue<string>("likelihood", 1);
		string outFileName = args.getValue<string>("likelihood", 2);
		int numIterations = args.getValue<int>("likelihood", 3);
		
		classifier.saveLikelihoods(testFileName, shypFileName, outFileName, numIterations);
	}
	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	
	
	// -------------------------------------------------------------------------
	
	void VJCascadeLearner::doPosteriors(const nor_utils::Args& args)
	{
		AdaBoostMHClassifier classifier(args, _verbose);
		int numofargs = args.getNumValues( "posteriors" );
		// -posteriors <dataFile> <shypFile> <outFile> <numIters>
		string testFileName = args.getValue<string>("posteriors", 0);
		string shypFileName = args.getValue<string>("posteriors", 1);
		string outFileName = args.getValue<string>("posteriors", 2);
		int numIterations = args.getValue<int>("posteriors", 3);
		int period = 0;
		
		if ( numofargs == 5 )
			period = args.getValue<int>("posteriors", 4);
		
		classifier.savePosteriors(testFileName, shypFileName, outFileName, numIterations, period);
	}
	
	// -------------------------------------------------------------------------
	
	void VJCascadeLearner::doROC(const nor_utils::Args& args)
	{
		AdaBoostMHClassifier classifier(args, _verbose);
		
		// -posteriors <dataFile> <shypFile> <outFileName>
		string testFileName = args.getValue<string>("roc", 0);
		string shypFileName = args.getValue<string>("roc", 1);
		string outFileName = args.getValue<string>("roc", 2);
		int numIterations = args.getValue<int>("roc", 3);
		
		classifier.saveROC(testFileName, shypFileName, outFileName, numIterations);
	}
	
	
	// -------------------------------------------------------------------------
	
	void VJCascadeLearner::doCalibratedPosteriors(const nor_utils::Args& args)
	{
		AdaBoostMHClassifier classifier(args, _verbose);
		
		// -posteriors <dataFile> <shypFile> <outFileName>
		string testFileName = args.getValue<string>("cposteriors", 0);
		string shypFileName = args.getValue<string>("cposteriors", 1);
		string outFileName = args.getValue<string>("cposteriors", 2);
		int numIterations = args.getValue<int>("cposteriors", 3);
		
		classifier.saveCalibratedPosteriors(testFileName, shypFileName, outFileName, numIterations);
	}
	
	
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	// -------------------------------------------------------------------------
	
	float VJCascadeLearner::updateWeights(InputData* pData, BaseLearner* pWeakHypothesis)
	{
		const int numExamples = pData->getNumExamples();
		const int numClasses = pData->getNumClasses();
		
		const float alpha = pWeakHypothesis->getAlpha();
		
		float Z = 0; // The normalization factor
		
		_hy.resize(numExamples);
		for ( int i = 0; i < numExamples; ++i) {
			_hy[i].resize(numClasses);
			fill( _hy[i].begin(), _hy[i].end(), 0.0 );
		}
		// recompute weights
		// computing the normalization factor Z
		
		// for each example
		for (int i = 0; i < numExamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;
			
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				_hy[i][lIt->idx] = pWeakHypothesis->classify(pData, i, lIt->idx) * // h_l(x_i)
				lIt->y;
				Z += lIt->weight * // w
				exp( 
					-alpha * _hy[i][lIt->idx] // -alpha * h_l(x_i) * y_i
					);
				// important!
				// _hy[i] must be a vector with different sizes, depending on the
				// example!
				// so it will become:
				// _hy[i][l] 
				// where l is NOT the index of the label (lIt->idx), but the index in the 
				// label vector of the example
			}
		}
		
		float gamma = 0;
		
		// Now do the actual re-weight
		// (and compute the edge at the same time)
		// for each example
		for (int i = 0; i < numExamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;
			
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				float w = lIt->weight;
				gamma += w * _hy[i][lIt->idx];
				//if ( gamma < -0.8 ) {
				//	cout << gamma << endl;
				//}
				// The new weight is  w * exp( -alpha * h(x_i) * y_i ) / Z
				lIt->weight = w * exp( -alpha * _hy[i][lIt->idx] ) / Z;
			}
		}
		
		
		//for (int i = 0; i < numExamples; ++i)
		//{
		//   for (int l = 0; l < numClasses; ++l)
		//   {
		//      _hy[i][l] = pWeakHypothesis->classify(pData, i, l) * // h_l(x_i)
		//                  pData->getLabel(i, l); // y_i
		
		//      Z += pData->getWeight(i, l) * // w
		//           exp( 
		//             -alpha * _hy[i][l] // -alpha * h_l(x_i) * y_i
		//           );
		//   } // numClasses
		//} // numExamples
		
		// The edge. It measures the
		// accuracy of the current weak hypothesis relative to random guessing
		
		//// Now do the actual re-weight
		//// (and compute the edge at the same time)
		//for (int i = 0; i < numExamples; ++i)
		//{
		//   for (int l = 0; l < numClasses; ++l)
		//   {  
		//      float w = pData->getWeight(i, l);
		
		//      gamma += w * _hy[i][l];
		
		//      // The new weight is  w * exp( -alpha * h(x_i) * y_i ) / Z
		//      pData->setWeight( i, l, 
		//                        w * exp( -alpha * _hy[i][l] ) / Z );
		//   } // numClasses
		//} // numExamples
		
		return gamma;
	}
	
	// -------------------------------------------------------------------------
	
	int VJCascadeLearner::resumeWeakLearners(InputData* pTrainingData)
	{
		if (_resumeShypFileName.empty())
			return 0;
		
		if (_verbose > 0)
			cout << "Reloading strong hypothesis file <" << _resumeShypFileName << ">.." << flush;
		
		// The class that loads the weak hypotheses
		UnSerialization us;
		
		// loads them stagewise
		for(int stagei=0; stagei < _numIterations; ++stagei )
		{		
			us.loadHypotheses(_resumeShypFileName, _foundHypotheses[stagei], pTrainingData, _verbose);
			
		}
		
		if (_verbose > 0)
			cout << "Done!" << endl;
		
		// return the number of iterations found
		return static_cast<int>( _foundHypotheses.size() );
	}
	
	// -------------------------------------------------------------------------
	
	void VJCascadeLearner::resetWeights(InputData* pData)
	{
		int numOfClasses = pData->getNumClasses();
		vector< int > numPerClasses(numOfClasses);
		int numOfSamples = pData->getNumExamples();
		vector< double > wi( numOfClasses );		
		
		fill(numPerClasses.begin(),numPerClasses.end(), 0 );
		for (int i = 0; i < numOfSamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;
			
			for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				if ( lIt->y > 0 )
					numPerClasses[lIt->idx]++;
			}
			
		}			
		
		
		// we assume pl = 1/K
		for( int i = 0; i < numOfClasses; i ++ ) {
			wi[i] =  1.0  / (2.0 * numPerClasses[i]);
		}
		//cout << endl;
		
		
		
		
		//this->_nExamplesPerClass
		// for each example
		for (int i = 0; i < numOfSamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;
			
			float sumPos = 0;
			float sumNeg = 0;
			
			// first find the sum of the weights					
			int i = 0;
			for ( lIt = labels.begin(); lIt != labels.end(); ++lIt, i++ )
			{
				lIt->weight = wi[lIt->idx];
			}
			
		}
		// check for the sum of weights!
		double sumWeight = 0;
		for (int i = 0; i < numOfSamples; ++i)
		{
			vector<Label>& labels = pData->getLabels(i);
			vector<Label>::iterator lIt;
			
			// first find the sum of the weights
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
				sumWeight += lIt->weight;
		}
		
		if ( !nor_utils::is_zero(sumWeight-1.0, 1E-3 ) )
		{
			cerr << "\nERROR: Sum of weights (" << sumWeight << ") != 1!" << endl;
			cerr << "Try a different weight policy (--weightpolicy under 'Basic Algorithm Options')!" << endl;
			//exit(1);
		}
		
	
	}
	
	// -------------------------------------------------------------------------
} // end of namespace MultiBoost
